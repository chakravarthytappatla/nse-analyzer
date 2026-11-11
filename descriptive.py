"""
==============================================================================
descriptive.py — PAGE 1 · Descriptive Statistics
==============================================================================
The original analysis (moments, beta regression, Sharpe / Treynor / Jensen /
Fama / Sortino, VaR, pivots, RSI, valuation read) — moved here from app.py
unchanged in method, with these fixes:
  * Net-asset-risk drawdown was computed on *returns* instead of *prices*
  * CAGR + Jarque-Bera normality test added to the descriptive block
  * charts use the thread-safe Figure API
==============================================================================
"""
import numpy as np
import pandas as pd
from scipy import stats

from core import (
    TRADING_DAYS_PER_YEAR, ACCENT, INFO, GAIN, LOSS, CHART_DIM,
    fetch_price_history_cached, new_figure, set_title, legend, fig_to_b64,
)

RSI_PERIOD = 14
DMA_WINDOW = 10
MAX_TICKERS = 6


# ==============================================================================
# STATISTICAL METRICS
# ==============================================================================

def compute_descriptive_stats(returns):
    clean = returns.dropna()
    mode_result = stats.mode(np.round(clean.values, 4), keepdims=True)
    mode_val = float(mode_result.mode[0]) if len(mode_result.mode) > 0 else float("nan")
    jb_stat, jb_p = stats.jarque_bera(clean)
    return {
        "mean_daily": float(clean.mean()),
        "median_daily": float(clean.median()),
        "mode_daily": mode_val,
        "skewness": float(stats.skew(clean)),
        "kurtosis": float(stats.kurtosis(clean)),
        "annual_return": float((1 + clean.mean()) ** TRADING_DAYS_PER_YEAR - 1),
        "annual_volatility": float(clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "jb_stat": float(jb_stat),
        "jb_p": float(jb_p),
    }


def compute_cagr(close):
    close = close.dropna()
    years = (close.index[-1] - close.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1)


def compute_beta_alpha_regression(stock_returns, market_returns):
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned.columns = ["stock", "market"]
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        aligned["market"], aligned["stock"]
    )
    return {
        "beta": float(slope),
        "alpha_daily_intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
    }


# ==============================================================================
# RISK & TECHNICAL INDICATORS
# ==============================================================================

def compute_max_drawdown(close_series):
    cumulative = (1 + close_series.pct_change().fillna(0)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def compute_dma(close_series, window=DMA_WINDOW):
    return close_series.rolling(window=window).mean()


def compute_rsi(close_series, period=RSI_PERIOD):
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_pivot_points(high, low, close):
    p = (high + low + close) / 3
    r1, s1 = 2 * p - low, 2 * p - high
    r2, s2 = p + (high - low), p - (high - low)
    r3, s3 = high + 2 * (p - low), low - 2 * (high - p)
    return {"P": p, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}


def classify_valuation(current_price, dma10, close_series):
    hist_min, hist_max = float(close_series.min()), float(close_series.max())
    percentile = (current_price - hist_min) / (hist_max - hist_min) if hist_max > hist_min else 0.5
    price_vs_dma_pct = (current_price - dma10) / dma10

    if price_vs_dma_pct < -0.02 and percentile < 0.55:
        return "Undervalued", percentile, price_vs_dma_pct
    elif price_vs_dma_pct > 0.02 and percentile > 0.75:
        return "Overvalued", percentile, price_vs_dma_pct
    return "Fair Value", percentile, price_vs_dma_pct


# ==============================================================================
# PORTFOLIO PERFORMANCE MEASURES
# ==============================================================================

def grade_sharpe(x):
    if x is None or not np.isfinite(x): return "N/A"
    if x > 1.0: return "Excellent"
    if x >= 0.75: return "Good"
    if x >= 0.50: return "Fair"
    return "Poor"


def grade_treynor(x):
    if x is None or not np.isfinite(x): return "N/A"
    if x > 0.25: return "Excellent"
    if x >= 0.15: return "Good"
    if x >= 0.10: return "Fair"
    return "Poor"


def grade_alpha(x):
    if x is None or not np.isfinite(x): return "N/A"
    if x > 6.0: return "Excellent"
    if x >= 3.0: return "Good"
    if x >= 0.0: return "Fair"
    return "Poor"


def grade_fama(x):
    if x is None or not np.isfinite(x): return "N/A"
    if x > 3.0: return "Excellent"
    if x >= 1.0: return "Good"
    if x >= 0.0: return "Fair"
    return "Poor"


def grade_sortino(x):
    if x is None or not np.isfinite(x): return "N/A"
    if x > 1.25: return "Excellent"
    if x >= 0.90: return "Good"
    if x >= 0.60: return "Fair"
    return "Poor"


def compute_sortino_ratio(daily_returns, annual_return, daily_rf, target_daily_return=0.0):
    clean = daily_returns.dropna()
    downside = clean[clean < target_daily_return]
    if len(downside) == 0:
        downside_deviation = 0.0
    else:
        downside_deviation = float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sortino = ((annual_return - daily_rf * TRADING_DAYS_PER_YEAR) / downside_deviation
               if downside_deviation != 0 else float("nan"))
    return sortino, downside_deviation


def compute_performance_measures(stock_stats, close, daily_returns, beta, annual_rf,
                                 market_return, market_vol):
    Rp, sigma_p = stock_stats["annual_return"], stock_stats["annual_volatility"]
    Rm, sigma_m = market_return, market_vol

    daily_rf = annual_rf / TRADING_DAYS_PER_YEAR
    clean_returns = daily_returns.dropna()
    daily_excess = clean_returns - daily_rf

    daily_std = clean_returns.std()
    sharpe = 0.0 if (daily_std == 0 or np.isnan(daily_std)) else float(
        (daily_excess.mean() / daily_std) * np.sqrt(TRADING_DAYS_PER_YEAR))

    treynor = (Rp - annual_rf) / beta if beta != 0 else float("nan")
    jensen_alpha = Rp - (annual_rf + beta * (Rm - annual_rf))

    if sigma_m != 0:
        required_return_total_risk = annual_rf + (sigma_p / sigma_m) * (Rm - annual_rf)
        fama_net_selectivity = (Rp - annual_rf) - (required_return_total_risk - annual_rf)
    else:
        fama_net_selectivity = float("nan")

    sortino, downside_dev = compute_sortino_ratio(clean_returns, Rp, daily_rf)

    var_95 = float(np.percentile(clean_returns, 5)) * 100
    # FIX: drawdown must be measured on the price path, not on returns.
    net_asset_risk_score = float(sigma_p * (abs(compute_max_drawdown(close)) + 1.0) * 100)

    return {
        "sharpe": sharpe, "sharpe_grade": grade_sharpe(sharpe),
        "treynor": treynor, "treynor_grade": grade_treynor(treynor),
        "jensen_alpha_pct": jensen_alpha * 100, "jensen_alpha_grade": grade_alpha(jensen_alpha * 100),
        "fama_pct": fama_net_selectivity * 100, "fama_grade": grade_fama(fama_net_selectivity * 100),
        "sortino": sortino, "sortino_grade": grade_sortino(sortino),
        "downside_deviation_pct": downside_dev * 100,
        "var_95_pct": var_95,
        "net_asset_risk_pct": net_asset_risk_score,
    }


# ==============================================================================
# CHARTS
# ==============================================================================

def render_returns_histogram(ticker, daily_returns):
    fig, ax = new_figure((9, 4.2))
    clean_pct = daily_returns.dropna() * 100
    n_bins = min(60, max(20, int(len(clean_pct) / 15)))
    _, bins, patches = ax.hist(clean_pct, bins=n_bins, edgecolor="#12161c", alpha=0.95)
    for patch, edge in zip(patches, bins[:-1]):
        patch.set_facecolor(LOSS if edge < 0 else GAIN)
    mean_val = float(clean_pct.mean())
    ax.axvline(mean_val, color=ACCENT, linestyle="--", linewidth=1.4, label=f"Mean {mean_val:.2f}%")
    ax.axvline(0, color=CHART_DIM, linestyle=":", linewidth=1)
    set_title(ax, f"{ticker} — Daily Return Distribution")
    ax.set_xlabel("Daily return (%)", fontsize=9)
    ax.set_ylabel("Frequency (trading days)", fontsize=9)
    legend(ax, "upper right")
    return fig_to_b64(fig)


def render_benchmark_comparison_chart(ticker, close, benchmark_close):
    aligned = pd.concat([close, benchmark_close], axis=1).dropna()
    aligned.columns = ["stock", "benchmark"]
    if aligned.empty:
        return None
    stock_growth = aligned["stock"] / aligned["stock"].iloc[0] * 100
    bench_growth = aligned["benchmark"] / aligned["benchmark"].iloc[0] * 100

    fig, ax = new_figure((9, 4.2))
    ax.plot(stock_growth.index, stock_growth.values, label=ticker, linewidth=1.7, color=ACCENT)
    ax.plot(bench_growth.index, bench_growth.values, label="Nifty 50", linewidth=1.4,
            color=INFO, linestyle="--")
    set_title(ax, f"{ticker} vs Nifty 50 — Growth of ₹100")
    ax.set_ylabel("Value of ₹100 invested", fontsize=9)
    legend(ax)
    return fig_to_b64(fig)


def render_price_chart(ticker, close, dma10, pivots):
    fig, ax = new_figure((9, 4.6))
    ax.plot(close.index, close.values, label="Close", linewidth=1.6, color=ACCENT)
    ax.plot(dma10.index, dma10.values, label=f"{DMA_WINDOW}-DMA", linewidth=1.2,
            color=INFO, linestyle="--")
    ax.axhline(pivots["S1"], color=GAIN, linestyle=":", linewidth=1.2, label=f"S1 ({pivots['S1']:.1f})")
    ax.axhline(pivots["R1"], color=LOSS, linestyle=":", linewidth=1.2, label=f"R1 ({pivots['R1']:.1f})")
    set_title(ax, ticker)
    legend(ax)
    return fig_to_b64(fig)


METRIC_EXPLANATIONS = {
    "price": "The most recent closing price on the NSE.",
    "dma10": "Average closing price over the last 10 trading days — a short-term trend line.",
    "rsi": "Momentum gauge from 0–100. Above 70 usually signals overbought, below 30 usually signals oversold.",
    "beta": "How much the stock tends to move for every 1% move in the Nifty 50. Above 1 means more volatile than the market; below 1 means less.",
    "r_squared": "Share of the stock's daily moves explained by the Nifty 50. Low R² = mostly stock-specific risk.",
    "annual_return": "Mean daily return compounded over 252 trading days (arithmetic annualisation).",
    "cagr": "Actual compound annual growth of the price over the 5-year window (geometric).",
    "annual_vol": "How much the stock's returns swing around their average each year — higher means a bumpier ride.",
    "max_drawdown": "The single worst peak-to-trough fall an investor would have lived through in the last 5 years.",
    "skewness": "Whether extreme returns lean positive or negative. Positive skew means occasional big gains; negative skew means occasional big losses.",
    "kurtosis": "Excess kurtosis: how often extreme moves happen versus a normal distribution (0). Higher means fatter tails.",
    "jarque_bera": "Normality test on daily returns. p < 0.05 means returns are NOT normal — so normal-based risk numbers understate tail risk.",
    "sharpe": "Return earned per unit of total risk taken, above the risk-free rate. Higher is better.",
    "treynor": "Return earned per unit of market risk (beta) taken, above the risk-free rate. Higher is better.",
    "alpha": "Return the stock generated beyond what its market risk alone would predict — a rough read on stock-picking skill.",
    "fama": "Extra return earned beyond what was needed to compensate for the stock's total risk — isolates true selectivity from just taking on more risk.",
    "sortino": "Like the Sharpe ratio, but only penalises downside swings, not upside ones — often a fairer risk measure for cautious investors.",
    "valuation": "A quick read on whether the price looks cheap or expensive relative to its own recent trend and 5-year range.",
    "outperformance": "How much better (or worse) the stock did than the Nifty 50 over the same period, annualised.",
    "pivots": "Classic support/resistance levels calculated from the most recent trading day's high, low and close.",
    "histogram": "Each bar is how many trading days had a return in that range — a wider, flatter spread means bumpier, riskier days.",
    "benchmark_chart": "Shows what ₹100 invested 5 years ago would be worth today in the stock versus in the Nifty 50 itself.",
    "var_95_pct": "95% one-day historical Value at Risk — on 1 day in 20 the loss is expected to be at least this large.",
    "net_asset_risk_pct": "Net Asset Risk score combining annual volatility and peak-to-trough drawdown magnitude.",
}


# ==============================================================================
# CORE ANALYSIS FOR A SINGLE TICKER
# ==============================================================================

def analyze_ticker(ticker, benchmark_close, benchmark_returns, market_return, market_vol, risk_free_rate):
    df = fetch_price_history_cached(ticker)
    if df.empty or "Close" not in df.columns or df["Close"].notna().sum() < 60:
        return {"ticker": ticker, "error": "No (or too little) price data found for this symbol."}

    close = df["Close"].dropna()
    daily_returns = close.pct_change()

    desc_stats = compute_descriptive_stats(daily_returns)
    reg_stats = compute_beta_alpha_regression(daily_returns, benchmark_returns)

    max_dd = compute_max_drawdown(close)
    dma10 = compute_dma(close)
    rsi = compute_rsi(close)

    current_price = float(close.iloc[-1])
    current_dma10 = float(dma10.iloc[-1])
    rsi_last = rsi.iloc[-1]
    current_rsi = float(rsi_last) if pd.notna(rsi_last) else None

    last = df.dropna(subset=["High", "Low", "Close"]).iloc[-1]
    pivots = compute_pivot_points(float(last["High"]), float(last["Low"]), float(last["Close"]))

    valuation, percentile, _ = classify_valuation(current_price, current_dma10, close)
    perf = compute_performance_measures(desc_stats, close, daily_returns, reg_stats["beta"],
                                        risk_free_rate, market_return, market_vol)
    outperformance_pct = (desc_stats["annual_return"] - market_return) * 100

    return {
        "ticker": ticker,
        "price": round(current_price, 2),
        "dma10": round(current_dma10, 2),
        "rsi": round(current_rsi, 1) if current_rsi is not None else None,
        "rsi_zone": ("Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30
                     else "Neutral") if current_rsi is not None else "N/A",
        "beta": round(reg_stats["beta"], 2),
        "r_squared": round(reg_stats["r_squared"], 3),
        "annual_return_pct": round(desc_stats["annual_return"] * 100, 2),
        "cagr_pct": round(compute_cagr(close) * 100, 2),
        "annual_vol_pct": round(desc_stats["annual_volatility"] * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "skewness": round(desc_stats["skewness"], 2),
        "kurtosis": round(desc_stats["kurtosis"], 2),
        "jb_p": desc_stats["jb_p"],
        "mean_daily_pct": round(desc_stats["mean_daily"] * 100, 4),
        "median_daily_pct": round(desc_stats["median_daily"] * 100, 4),
        "mode_daily_pct": round(desc_stats["mode_daily"] * 100, 4),
        "sharpe": round(perf["sharpe"], 2), "sharpe_grade": perf["sharpe_grade"],
        "treynor": round(perf["treynor"], 3), "treynor_grade": perf["treynor_grade"],
        "alpha_pct": round(perf["jensen_alpha_pct"], 2), "alpha_grade": perf["jensen_alpha_grade"],
        "fama_pct": round(perf["fama_pct"], 2), "fama_grade": perf["fama_grade"],
        "sortino": round(perf["sortino"], 2) if pd.notna(perf["sortino"]) else None,
        "sortino_grade": perf["sortino_grade"],
        "downside_deviation_pct": round(perf["downside_deviation_pct"], 2),
        "var_95_pct": round(perf["var_95_pct"], 2),
        "net_asset_risk_pct": round(perf["net_asset_risk_pct"], 2),
        "benchmark_annual_return_pct": round(market_return * 100, 2),
        "outperformance_pct": round(outperformance_pct, 2),
        "valuation": valuation,
        "price_percentile": round(percentile * 100, 1),
        "pivots": {k: round(v, 2) for k, v in pivots.items()},
        "chart_b64": render_price_chart(ticker, close, dma10, pivots),
        "histogram_b64": render_returns_histogram(ticker, daily_returns),
        "benchmark_chart_b64": render_benchmark_comparison_chart(ticker, close, benchmark_close),
        "_returns": daily_returns,
    }


def run_descriptive(symbols, ctx):
    """Page-1 payload. Market stats use the same arithmetic annualisation as
    each stock so Jensen / Fama / outperformance compare like with like."""
    bench_close = ctx["bench_close"]
    bench_returns = bench_close.pct_change()
    mstats = compute_descriptive_stats(bench_returns)
    market_return, market_vol = mstats["annual_return"], mstats["annual_volatility"]

    results, returns_for_corr = [], {}
    for symbol in symbols:
        try:
            r = analyze_ticker(symbol, bench_close, bench_returns, market_return, market_vol, ctx["rf"])
        except Exception as exc:  # one bad symbol must not sink the batch
            r = {"ticker": symbol, "error": f"Analysis failed: {exc}"}
        if "_returns" in r:
            returns_for_corr[symbol] = r.pop("_returns")
        results.append(r)

    correlation = None
    if len(returns_for_corr) > 1:
        corr_df = pd.DataFrame(returns_for_corr).dropna().corr().round(3)
        correlation = {"tickers": list(corr_df.columns), "matrix": corr_df.values.tolist()}

    return {
        "market_annual_return_pct": round(market_return * 100, 2),
        "market_annual_vol_pct": round(market_vol * 100, 2),
        "results": results,
        "correlation": correlation,
        "explanations": METRIC_EXPLANATIONS,
    }
