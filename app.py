"""
==============================================================================
NSE TERMINAL — Flask backend
==============================================================================
Serves:
  GET  /                     -> the web UI
  GET  /api/suggest?q=TC     -> ticker autocomplete (Yahoo Finance search,
                                 filtered to NSE symbols, ".NS" is implicit)
  POST /api/analyze          -> runs the full quant/macro/technical analysis
                                 for the requested NSE symbols and returns JSON
==============================================================================
"""
import os
import base64
import contextlib
import io
import logging
import time
import warnings
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")  # headless rendering for server-side chart generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yh
from flask import Flask, jsonify, render_template, request
from scipy import stats

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

app = Flask(__name__)

# ==============================================================================
# CONFIG
# ==============================================================================

BENCHMARK_TICKER = "^NSEI"
BOND_YIELD_TICKER_CANDIDATES = ["^IN10YT=X", "IN10Y=X", "^IN10Y"]
FALLBACK_RISK_FREE_RATE = 0.070

TRADING_DAYS_PER_YEAR = 252
RSI_PERIOD = 14
DMA_WINDOW = 10
MAX_TICKERS_PER_REQUEST = 6

CACHE_TTL_SECONDS = 15 * 60  # 15 minutes — avoids hammering Yahoo on every click
_cache = {}


def cache_get(key):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL_SECONDS:
        return entry["value"]
    return None


def cache_set(key, value):
    _cache[key] = {"value": value, "ts": time.time()}


def date_range():
    end = datetime.today()
    start = end - timedelta(days=5 * 365)
    return start, end


# ==============================================================================
# DATA ACQUISITION
# ==============================================================================

def fetch_price_history(ticker, start, end):
    """Download OHLCV data for a single ticker, with yfinance's noisy
    stdout/stderr error prints suppressed."""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        df = yh.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


def fetch_price_history_cached(ticker):
    key = f"price:{ticker}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    start, end = date_range()
    df = fetch_price_history(ticker, start, end)
    cache_set(key, df)
    return df


def fetch_dynamic_risk_free_rate():
    key = "risk_free_rate"
    cached = cache_get(key)
    if cached is not None:
        return cached

    start, end = date_range()
    for candidate in BOND_YIELD_TICKER_CANDIDATES:
        try:
            bond_df = fetch_price_history(candidate, start, end)
            if bond_df.empty or "Close" not in bond_df.columns:
                continue
            latest_yield = float(bond_df["Close"].dropna().iloc[-1])
            result = (latest_yield / 100.0, bond_df, candidate)
            cache_set(key, result)
            return result
        except Exception:
            continue

    result = (FALLBACK_RISK_FREE_RATE, pd.DataFrame(), None)
    cache_set(key, result)
    return result


def macro_yield_commentary(bond_df):
    if bond_df.empty or "Close" not in bond_df.columns or len(bond_df) < 30:
        return "NEUTRAL", ("Insufficient bond-yield history is available right now to "
                            "assess the macro backdrop, so a neutral stance is assumed.")

    yields = bond_df["Close"].dropna()
    start_yield = float(yields.iloc[0])
    end_yield = float(yields.iloc[-1])
    change_bps = (end_yield - start_yield) * 100

    recent_window = yields.tail(126) if len(yields) >= 126 else yields
    recent_slope = float(recent_window.iloc[-1] - recent_window.iloc[0])

    if change_bps <= -50 or recent_slope <= -0.25:
        stance = "FAVORABLE"
        note = (f"Bond yields have declined roughly {abs(change_bps):.0f} bps over the "
                f"5-year window ({start_yield:.2f}% \u2192 {end_yield:.2f}%). Falling yields "
                f"lower the discount rate on future cash flows, which tends to support "
                f"and expand equity valuations.")
    elif change_bps >= 100 or recent_slope >= 0.5:
        stance = "UNFAVORABLE"
        note = (f"Bond yields have risen sharply, roughly {change_bps:.0f} bps over the "
                f"5-year window ({start_yield:.2f}% \u2192 {end_yield:.2f}%). Sharp yield "
                f"spikes signal tightening conditions and typically compress equity "
                f"valuation multiples.")
    else:
        stance = "NEUTRAL"
        note = (f"Bond yields have moved moderately, roughly {change_bps:.0f} bps over the "
                f"5-year window ({start_yield:.2f}% \u2192 {end_yield:.2f}%). This is a "
                f"broadly stable-rate backdrop for equities.")

    return stance, note


# ==============================================================================
# STATISTICAL METRICS
# ==============================================================================

def compute_descriptive_stats(returns):
    clean = returns.dropna()
    mode_result = stats.mode(np.round(clean.values, 4), keepdims=True)
    mode_val = float(mode_result.mode[0]) if len(mode_result.mode) > 0 else float("nan")
    return {
        "mean_daily": float(clean.mean()),
        "median_daily": float(clean.median()),
        "mode_daily": mode_val,
        "skewness": float(stats.skew(clean)),
        "kurtosis": float(stats.kurtosis(clean)),
        "annual_return": float((1 + clean.mean()) ** TRADING_DAYS_PER_YEAR - 1),
        "annual_volatility": float(clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
    }


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
    if x > 1.0: return "Excellent"
    if x >= 0.75: return "Good"
    if x >= 0.50: return "Fair"
    return "Poor"


def grade_treynor(x):
    if x > 0.25: return "Excellent"
    if x >= 0.15: return "Good"
    if x >= 0.10: return "Fair"
    return "Poor"


def grade_alpha(x):
    if x > 6.0: return "Excellent"
    if x >= 3.0: return "Good"
    if x >= 0.0: return "Fair"
    return "Poor"


def grade_fama(x):
    if x > 3.0: return "Excellent"
    if x >= 1.0: return "Good"
    if x >= 0.0: return "Fair"
    return "Poor"


def compute_performance_measures(stock_stats, beta, rf, market_return, market_vol):
    Rp, sigma_p = stock_stats["annual_return"], stock_stats["annual_volatility"]
    Rm, sigma_m = market_return, market_vol

    sharpe = (Rp - rf) / sigma_p if sigma_p != 0 else float("nan")
    treynor = (Rp - rf) / beta if beta != 0 else float("nan")
    jensen_alpha = Rp - (rf + beta * (Rm - rf))

    required_return_total_risk = rf + (sigma_p / sigma_m) * (Rm - rf) if sigma_m != 0 else float("nan")
    fama_net_selectivity = (Rp - rf) - (required_return_total_risk - rf) if sigma_m != 0 else float("nan")

    return {
        "sharpe": sharpe, "sharpe_grade": grade_sharpe(sharpe),
        "treynor": treynor, "treynor_grade": grade_treynor(treynor),
        "jensen_alpha_pct": jensen_alpha * 100, "jensen_alpha_grade": grade_alpha(jensen_alpha * 100),
        "fama_pct": fama_net_selectivity * 100, "fama_grade": grade_fama(fama_net_selectivity * 100),
    }


# ==============================================================================
# CHART RENDERING (server-side, returned as base64 PNG so the frontend can
# simply drop it into an <img> tag)
# ==============================================================================

def render_price_chart(ticker, close, dma10, pivots):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
    fig.patch.set_facecolor("#12161c")
    ax.set_facecolor("#12161c")

    ax.plot(close.index, close.values, label="Close", linewidth=1.6, color="#ffb347")
    ax.plot(dma10.index, dma10.values, label=f"{DMA_WINDOW}-DMA", linewidth=1.2,
            color="#5ecbf1", linestyle="--")
    ax.axhline(pivots["S1"], color="#37d67a", linestyle=":", linewidth=1.2,
               label=f"S1 ({pivots['S1']:.1f})")
    ax.axhline(pivots["R1"], color="#ff5c5c", linestyle=":", linewidth=1.2,
               label=f"R1 ({pivots['R1']:.1f})")

    ax.set_title(ticker, fontsize=13, color="#f2f2f2", weight="bold", loc="left")
    ax.tick_params(colors="#9aa4b2", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2a313c")
    ax.grid(alpha=0.15, color="#5a6577")
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor="#e6e6e6")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ==============================================================================
# CORE ANALYSIS FOR A SINGLE TICKER
# ==============================================================================

def analyze_ticker(ticker, benchmark_returns, market_return, market_vol, risk_free_rate):
    df = fetch_price_history_cached(ticker)
    if df.empty or "Close" not in df.columns:
        return {"ticker": ticker, "error": "No data found for this symbol."}

    close = df["Close"]
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

    last_high, last_low, last_close = float(df["High"].iloc[-1]), float(df["Low"].iloc[-1]), float(df["Close"].iloc[-1])
    pivots = compute_pivot_points(last_high, last_low, last_close)

    valuation, percentile, price_vs_dma = classify_valuation(current_price, current_dma10, close)
    perf = compute_performance_measures(desc_stats, reg_stats["beta"], risk_free_rate, market_return, market_vol)
    chart_b64 = render_price_chart(ticker, close, dma10, pivots)

    return {
        "ticker": ticker,
        "price": round(current_price, 2),
        "dma10": round(current_dma10, 2),
        "rsi": round(current_rsi, 1) if current_rsi is not None else None,
        "rsi_zone": ("Overbought" if current_rsi and current_rsi > 70
                      else "Oversold" if current_rsi and current_rsi < 30
                      else "Neutral") if current_rsi is not None else "N/A",
        "beta": round(reg_stats["beta"], 2),
        "r_squared": round(reg_stats["r_squared"], 3),
        "annual_return_pct": round(desc_stats["annual_return"] * 100, 2),
        "annual_vol_pct": round(desc_stats["annual_volatility"] * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "skewness": round(desc_stats["skewness"], 2),
        "kurtosis": round(desc_stats["kurtosis"], 2),
        "mean_daily_pct": round(desc_stats["mean_daily"] * 100, 4),
        "median_daily_pct": round(desc_stats["median_daily"] * 100, 4),
        "mode_daily_pct": round(desc_stats["mode_daily"] * 100, 4),
        "sharpe": round(perf["sharpe"], 2), "sharpe_grade": perf["sharpe_grade"],
        "treynor": round(perf["treynor"], 3), "treynor_grade": perf["treynor_grade"],
        "alpha_pct": round(perf["jensen_alpha_pct"], 2), "alpha_grade": perf["jensen_alpha_grade"],
        "fama_pct": round(perf["fama_pct"], 2), "fama_grade": perf["fama_grade"],
        "valuation": valuation,
        "price_percentile": round(percentile * 100, 1),
        "pivots": {k: round(v, 2) for k, v in pivots.items()},
        "chart_b64": chart_b64,
        "_returns": daily_returns,  # kept only for correlation matrix, stripped before JSON
    }


# ==============================================================================
# ROUTES
# ==============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/suggest")
def api_suggest():
    """Autocomplete: typing 'TC' should surface 'TCS.NS' etc.
    Queries Yahoo Finance's own search endpoint and filters to NSE symbols."""
    q = request.args.get("q", "").strip()
    if len(q) < 1:
        return jsonify([])

    key = f"suggest:{q.lower()}"
    cached = cache_get(key)
    if cached is not None:
        return jsonify(cached)

    try:
        resp = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": q, "quotesCount": 10, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        data = resp.json()
        quotes = data.get("quotes", [])
    except Exception:
        quotes = []

    results = []
    seen = set()
    for item in quotes:
        symbol = item.get("symbol", "")
        exch = item.get("exchange", "")
        if not symbol.endswith(".NS") and exch not in ("NSI",):
            continue
        if not symbol.endswith(".NS"):
            continue
        name = item.get("longname") or item.get("shortname") or symbol
        if symbol in seen:
            continue
        seen.add(symbol)
        results.append({"symbol": symbol, "name": name})

    cache_set(key, results)
    return jsonify(results)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    payload = request.get_json(silent=True) or {}
    raw_symbols = payload.get("symbols", [])

    symbols = []
    for s in raw_symbols:
        s = (s or "").strip().upper()
        if not s:
            continue
        if not s.endswith(".NS"):
            s = f"{s}.NS"
        if s not in symbols:
            symbols.append(s)
    symbols = symbols[:MAX_TICKERS_PER_REQUEST]

    if not symbols:
        return jsonify({"error": "No valid symbols supplied."}), 400

    risk_free_rate, bond_df, bond_source = fetch_dynamic_risk_free_rate()
    macro_stance, macro_note = macro_yield_commentary(bond_df)

    benchmark_key = "benchmark_df"
    benchmark_df = cache_get(benchmark_key)
    if benchmark_df is None:
        start, end = date_range()
        benchmark_df = fetch_price_history(BENCHMARK_TICKER, start, end)
        cache_set(benchmark_key, benchmark_df)

    if benchmark_df.empty or "Close" not in benchmark_df.columns:
        return jsonify({"error": "Could not reach Yahoo Finance for Nifty 50 benchmark "
                                  "data right now. Please try again shortly."}), 502

    benchmark_returns = benchmark_df["Close"].pct_change()
    market_stats = compute_descriptive_stats(benchmark_returns)
    market_return, market_vol = market_stats["annual_return"], market_stats["annual_volatility"]

    results = []
    returns_for_corr = {}
    for symbol in symbols:
        r = analyze_ticker(symbol, benchmark_returns, market_return, market_vol, risk_free_rate)
        if "error" not in r:
            returns_for_corr[symbol] = r.pop("_returns")
        results.append(r)

    correlation = None
    if len(returns_for_corr) > 1:
        corr_df = pd.DataFrame(returns_for_corr).dropna().corr().round(3)
        correlation = {
            "tickers": list(corr_df.columns),
            "matrix": corr_df.values.tolist(),
        }

    return jsonify({
        "risk_free_rate_pct": round(risk_free_rate * 100, 2),
        "bond_source": bond_source,
        "macro_stance": macro_stance,
        "macro_note": macro_note,
        "market_annual_return_pct": round(market_return * 100, 2),
        "market_annual_vol_pct": round(market_vol * 100, 2),
        "results": results,
        "correlation": correlation,
    })
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8790))
    app.run(host='0.0.0.0', port=port)