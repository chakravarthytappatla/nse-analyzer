"""
==============================================================================
portfolio.py — PAGE 3 · Portfolio Management Analysis
==============================================================================
  * Covariance: Ledoit-Wolf shrinkage (more stable than the sample matrix)
  * Six allocation models, all long-only with a per-stock cap:
        Equal weight · Inverse volatility · Minimum variance · Risk parity (ERC)
        Max Sharpe (historical μ) · Max Sharpe (CAPM μ)
  * Model selection by WALK-FORWARD BACKTEST: re-optimise every quarter on the
    trailing 2 years, hold the next quarter out-of-sample, charge 10 bps per
    unit of turnover. The model with the best out-of-sample Sharpe wins.
  * Efficient frontier + Capital Market Line
  * Risk: 1-day VaR / CVaR at 95 % and 99 % (historical, parametric normal,
    Cornish-Fisher), risk contribution per stock, diversification ratio
  * 1-year Monte Carlo (stationary block bootstrap) → outcome percentiles
==============================================================================
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from core import (
    TRADING_DAYS_PER_YEAR, SERIES_COLORS, ACCENT, CHART_DIM, CHART_TEXT, INFO,
    adjusted_close, fetch_price_history_cached, fig_to_b64, legend, new_figure, set_title,
)

MIN_TICKERS, MAX_TICKERS = 2, 10
LOOKBACK_DAYS = 504          # 2-year estimation window
REBALANCE_DAYS = 63          # quarterly
COST_BPS = 10
MC_PATHS = 5000
MC_HORIZON = TRADING_DAYS_PER_YEAR
MC_BLOCK = 10
DEFAULT_MAX_WEIGHT = 0.40
DEFAULT_ERP = 0.0725

STRATEGIES = ["Equal weight", "Inverse volatility", "Minimum variance",
              "Risk parity (ERC)", "Max Sharpe (historical μ)", "Max Sharpe (CAPM μ)"]

EXPLANATIONS = {
    "backtest": "Each model is re-fitted every quarter on the previous 2 years only, then judged on the next quarter it has never seen. This is the honest way to pick a model — in-sample Sharpe always flatters optimisers.",
    "frontier": "Every point on the curve is the lowest-risk mix for a given return. The tangent line from the risk-free rate (CML) touches the max-Sharpe portfolio.",
    "var": "Value at Risk: the 1-day loss exceeded on only 5% (or 1%) of days. CVaR (Expected Shortfall) is the average loss on those worst days — Basel III's preferred measure.",
    "cornish_fisher": "Cornish-Fisher VaR adjusts the normal VaR for skew and fat tails, so it is usually closer to the historical number for equities.",
    "risk_contribution": "How much of total portfolio volatility each stock is responsible for. Weight ≠ risk — a 20% position can carry 35% of the risk.",
    "monte_carlo": "5,000 simulated one-year paths built by re-sampling 10-day blocks of real joint returns — keeps fat tails, volatility clusters and correlations intact.",
    "ledoit_wolf": "Ledoit-Wolf shrinks the noisy sample covariance toward a structured target, which cuts estimation error and makes optimised weights far more stable.",
}


# ==============================================================================
# ESTIMATORS
# ==============================================================================

def ledoit_wolf(returns):
    """Ledoit-Wolf (2004) shrinkage toward a scaled identity. Returns the
    daily covariance matrix and the shrinkage intensity."""
    X = returns.values - returns.values.mean(axis=0)
    T, N = X.shape
    S = X.T @ X / T
    mu = np.trace(S) / N
    F = mu * np.eye(N)
    delta = np.sum((S - F) ** 2)
    beta = (np.sum(np.sum(X ** 2, axis=1) ** 2) - T * np.sum(S ** 2)) / T ** 2
    beta = min(beta, delta)
    shrink = beta / delta if delta > 0 else 1.0
    return shrink * F + (1 - shrink) * S, float(shrink)


def capm_expected_returns(returns, market_returns, rf, erp):
    aligned = pd.concat([returns, market_returns.rename("_mkt")], axis=1).dropna()
    m = aligned["_mkt"]
    betas = aligned[returns.columns].apply(lambda s: s.cov(m) / m.var())
    return rf + betas.values * erp, betas.values


# ==============================================================================
# OPTIMISERS (long-only, sum to 1, per-asset cap)
# ==============================================================================

def _solve(objective, n, cap, x0=None):
    x0 = np.full(n, 1 / n) if x0 is None else x0
    res = minimize(objective, x0, method="SLSQP", bounds=[(0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x if res.success else x0, 0, cap)
    return w / w.sum()


def _cap_weights(w, cap):
    """Project raw weights onto {0 ≤ w ≤ cap, Σw = 1} by iterative capping."""
    w = w / w.sum()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def weights_for(strategy, cov, mu_hist, mu_capm, rf, cap):
    n = cov.shape[0]
    vol = np.sqrt(np.diag(cov))
    if strategy == "Equal weight":
        return np.full(n, 1 / n)
    if strategy == "Inverse volatility":
        return _cap_weights(1 / vol, cap)
    if strategy == "Minimum variance":
        return _solve(lambda w: w @ cov @ w * 1e4, n, cap)
    if strategy == "Risk parity (ERC)":
        def obj(w):
            port_var = w @ cov @ w
            rc = w * (cov @ w) / port_var
            return np.sum((rc - 1 / n) ** 2) * 1e3
        return _solve(obj, n, cap, _cap_weights(1 / vol, cap))
    mu = mu_hist if strategy == "Max Sharpe (historical μ)" else mu_capm
    def neg_sharpe(w):
        sd = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf) / sd if sd > 0 else 0.0
    return _solve(neg_sharpe, n, cap)


def estimate(window, market_window, rf, erp):
    cov_d, shrink = ledoit_wolf(window)
    cov = cov_d * TRADING_DAYS_PER_YEAR
    mu_hist = window.mean().values * TRADING_DAYS_PER_YEAR
    mu_capm, betas = capm_expected_returns(window, market_window, rf, erp)
    return cov, mu_hist, mu_capm, betas, shrink


# ==============================================================================
# PERFORMANCE METRICS
# ==============================================================================

def perf_stats(daily, rf, bench=None):
    daily = daily.dropna()
    n = len(daily)
    if n < 20:
        return {}
    growth = float((1 + daily).prod())
    years = n / TRADING_DAYS_PER_YEAR
    cagr = growth ** (1 / years) - 1
    vol = float(daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    rf_d = rf / TRADING_DAYS_PER_YEAR
    excess = daily - rf_d
    sharpe = float(excess.mean() / daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if daily.std() > 0 else None
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino = float(excess.mean() * TRADING_DAYS_PER_YEAR / downside) if downside > 0 else None
    curve = (1 + daily).cumprod()
    mdd = float((curve / curve.cummax() - 1).min())
    out = {"cagr": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino, "max_dd": mdd,
           "calmar": cagr / abs(mdd) if mdd < 0 else None}
    if bench is not None:
        a = pd.concat([daily, bench], axis=1).dropna()
        a.columns = ["p", "b"]
        beta = float(a["p"].cov(a["b"]) / a["b"].var())
        active = a["p"] - a["b"]
        te = float(active.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        bench_ann = float(a["b"].mean() * TRADING_DAYS_PER_YEAR)
        port_ann = float(a["p"].mean() * TRADING_DAYS_PER_YEAR)
        out.update({
            "beta": beta,
            "jensen_alpha": port_ann - (rf + beta * (bench_ann - rf)),
            "tracking_error": te,
            "info_ratio": float(active.mean() * TRADING_DAYS_PER_YEAR / te) if te > 0 else None,
        })
    return out


# ==============================================================================
# WALK-FORWARD BACKTEST
# ==============================================================================

def walk_forward(returns, market, rf, erp, cap):
    T, n = returns.shape
    daily = {s: [] for s in STRATEGIES}
    turnover = {s: [] for s in STRATEGIES}
    held = {s: np.zeros(n) for s in STRATEGIES}
    dates = []
    cost = COST_BPS / 10_000
    for start in range(LOOKBACK_DAYS, T, REBALANCE_DAYS):
        window = returns.iloc[start - LOOKBACK_DAYS:start]
        cov, mu_h, mu_c, _, _ = estimate(window, market.loc[window.index], rf, erp)
        hold = returns.iloc[start:start + REBALANCE_DAYS]
        growth = (1 + hold).cumprod().values
        dates.extend(hold.index)
        for s in STRATEGIES:
            w = weights_for(s, cov, mu_h, mu_c, rf, cap)
            to = float(np.abs(w - held[s]).sum()) if start > LOOKBACK_DAYS else 1.0
            turnover[s].append(to)
            value = growth @ w
            path = np.concatenate([[1.0], value])
            r = path[1:] / path[:-1] - 1
            r[0] -= to * cost
            daily[s].extend(r)
            drift = w * growth[-1]
            held[s] = drift / drift.sum()
    idx = pd.DatetimeIndex(dates)
    series = {s: pd.Series(daily[s], index=idx) for s in STRATEGIES}
    # first rebalance is the initial purchase, not turnover
    avg_turn = {s: float(np.mean(turnover[s][1:])) if len(turnover[s]) > 1 else 0.0 for s in STRATEGIES}
    return series, avg_turn


# ==============================================================================
# RISK
# ==============================================================================

def var_table(port_daily):
    r = port_daily.dropna().values
    mu, sd = r.mean(), r.std(ddof=1)
    s, k = stats.skew(r), stats.kurtosis(r)
    rows = []
    for conf in (0.95, 0.99):
        a = 1 - conf
        z = stats.norm.ppf(a)
        z_cf = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
                - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
        hist_var = -np.percentile(r, a * 100)
        hist_cvar = -r[r <= -hist_var].mean()
        norm_var = -(mu + z * sd)
        norm_cvar = -(mu - sd * stats.norm.pdf(z) / a)
        rows.append({"confidence": conf,
                     "historical_var": hist_var, "historical_cvar": hist_cvar,
                     "normal_var": norm_var, "normal_cvar": norm_cvar,
                     "cornish_fisher_var": -(mu + z_cf * sd)})
    return rows, float(s), float(k)


def monte_carlo(returns, w, amount, seed=42):
    rng = np.random.default_rng(seed)
    R = returns.values
    T = len(R)
    n_blocks = int(np.ceil(MC_HORIZON / MC_BLOCK))
    starts = rng.integers(0, T - MC_BLOCK, size=(MC_PATHS, n_blocks))
    offsets = np.arange(MC_BLOCK)
    idx = (starts[:, :, None] + offsets).reshape(MC_PATHS, -1)[:, :MC_HORIZON]
    port_r = R[idx] @ w                     # (paths, horizon)
    paths = amount * np.cumprod(1 + port_r, axis=1)
    final = paths[:, -1]
    pct = {p: np.percentile(paths, p, axis=0) for p in (5, 25, 50, 75, 95)}
    return {
        "paths_pct": pct, "final": final,
        "summary": {
            "expected": float(final.mean()), "median": float(np.median(final)),
            "p5": float(np.percentile(final, 5)), "p95": float(np.percentile(final, 95)),
            "prob_loss": float((final < amount).mean()),
            "var_1y_95": float(amount - np.percentile(final, 5)),
            "cvar_1y_95": float(amount - final[final <= np.percentile(final, 5)].mean()),
        },
    }


# ==============================================================================
# CHARTS
# ==============================================================================

def efficient_frontier(cov, mu, cap, n_points=40):
    n = len(mu)
    w_min = weights_for("Minimum variance", cov, mu, mu, 0, cap)
    r_min = float(w_min @ mu)
    # highest attainable return under the cap: fill best assets up to cap
    order, w_max, left = np.argsort(-mu), np.zeros(n), 1.0
    for i in order:
        take = min(cap, left)
        w_max[i], left = take, left - take
        if left <= 1e-12:
            break
    r_max = float(w_max @ mu)
    pts = []
    for target in np.linspace(r_min, r_max, n_points):
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
        res = minimize(lambda w: w @ cov @ w * 1e4, np.full(n, 1 / n), method="SLSQP",
                       bounds=[(0, cap)] * n, constraints=cons, options={"maxiter": 300})
        if res.success:
            pts.append((float(np.sqrt(res.x @ cov @ res.x)), float(target)))
    return pts


def render_frontier_chart(frontier, assets, strategy_points, rf, best_name):
    fig, ax = new_figure((10, 5.2))
    if frontier:
        fx, fy = zip(*frontier)
        ax.plot(np.array(fx) * 100, np.array(fy) * 100, color=CHART_TEXT, linewidth=2,
                label="Efficient frontier")
    ms = max(strategy_points, key=lambda p: (p["ret"] - rf) / p["vol"] if p["vol"] > 0 else -1e9)
    slope = (ms["ret"] - rf) / ms["vol"]
    xmax = max([a["vol"] for a in assets] + [p["vol"] for p in strategy_points]) * 1.1
    ax.plot([0, xmax * 100], [rf * 100, (rf + slope * xmax) * 100], color=CHART_DIM,
            linestyle="--", linewidth=1.2, label=f"Capital Market Line (Rf {rf*100:.2f}%)")
    for a in assets:
        ax.scatter(a["vol"] * 100, a["ret"] * 100, s=40, color=CHART_DIM, edgecolors="#12161c",
                   linewidths=2, zorder=3)
        ax.annotate(a["ticker"].replace(".NS", ""), (a["vol"] * 100, a["ret"] * 100),
                    xytext=(5, 4), textcoords="offset points", fontsize=7.5, color=CHART_DIM)
    for i, p in enumerate(strategy_points):
        is_best = p["name"] == best_name
        ax.scatter(p["vol"] * 100, p["ret"] * 100, s=140 if is_best else 70,
                   marker="*" if is_best else "o", color=SERIES_COLORS[i], edgecolors="#12161c",
                   linewidths=2, zorder=4, label=p["name"] + ("  ★ selected" if is_best else ""))
    set_title(ax, "Efficient frontier (2-yr window, Ledoit-Wolf Σ)")
    ax.set_xlabel("Annualised volatility (%)", fontsize=9)
    ax.set_ylabel("Annualised expected return (%)", fontsize=9)
    ax.set_xlim(left=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False, labelcolor="#e6e6e6")
    return fig_to_b64(fig)


def render_backtest_chart(series, bench, best_name):
    fig, ax = new_figure((9, 4.6))
    for i, (name, s) in enumerate(series.items()):
        curve = (1 + s).cumprod() * 100
        ax.plot(curve.index, curve.values, color=SERIES_COLORS[i],
                linewidth=2.4 if name == best_name else 1.3,
                alpha=1 if name == best_name else 0.75, label=name)
    b = bench.reindex(next(iter(series.values())).index).fillna(0)
    bc = (1 + b).cumprod() * 100
    ax.plot(bc.index, bc.values, color=CHART_DIM, linestyle="--", linewidth=1.3, label="Nifty 50")
    set_title(ax, "Walk-forward backtest — growth of ₹100 (out-of-sample, after costs)")
    ax.set_ylabel("Value of ₹100", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8,
              frameon=False, labelcolor="#e6e6e6")
    return fig_to_b64(fig)


def render_mc_chart(mc, amount):
    pct = mc["paths_pct"]
    days = np.arange(1, MC_HORIZON + 1)
    fig, ax = new_figure((9, 4.4))
    ax.fill_between(days, pct[5] / 1e5, pct[95] / 1e5, color=INFO, alpha=0.15, linewidth=0,
                    label="5th–95th percentile")
    ax.fill_between(days, pct[25] / 1e5, pct[75] / 1e5, color=INFO, alpha=0.30, linewidth=0,
                    label="25th–75th percentile")
    ax.plot(days, pct[50] / 1e5, color=ACCENT, linewidth=2, label="Median path")
    ax.axhline(amount / 1e5, color=CHART_DIM, linestyle=":", linewidth=1, label="Amount invested")
    set_title(ax, "Monte Carlo — 1-year outlook for the selected portfolio")
    ax.set_xlabel("Trading days ahead", fontsize=9)
    ax.set_ylabel("Portfolio value (₹ lakh)", fontsize=9)
    legend(ax, "upper left")
    return fig_to_b64(fig)


# ==============================================================================
# ORCHESTRATION
# ==============================================================================

def load_returns(symbols):
    closes, errors = {}, []
    for s in symbols:
        df = fetch_price_history_cached(s)
        if df.empty or "Close" not in df.columns:
            errors.append(f"{s}: no price data")
            continue
        closes[s] = adjusted_close(df)
    prices = pd.DataFrame(closes).dropna()
    return prices, errors


def run_portfolio(symbols, ctx, max_weight=DEFAULT_MAX_WEIGHT, amount=1_000_000, erp=DEFAULT_ERP):
    prices, errors = load_returns(symbols)
    tickers = list(prices.columns)
    if len(tickers) < MIN_TICKERS:
        return {"error": "Need at least 2 stocks with price data. " + "; ".join(errors)}
    min_needed = LOOKBACK_DAYS + 2 * REBALANCE_DAYS
    if len(prices) < min_needed:
        short = min(symbols, key=lambda s: len(fetch_price_history_cached(s)))
        return {"error": f"Only {len(prices)} overlapping trading days (need {min_needed}). "
                         f"{short} has the shortest history — remove it or pick older listings."}

    n = len(tickers)
    cap_used = max(max_weight, 1 / n)
    rf = ctx["rf"]
    returns = prices.pct_change().dropna()
    market = ctx["bench_returns"].reindex(returns.index).fillna(0)

    # --- model selection: out-of-sample backtest -----------------------------
    series, avg_turn = walk_forward(returns, market, rf, erp, cap_used)
    bench_oos = market.reindex(next(iter(series.values())).index)
    leaderboard = []
    for s in STRATEGIES:
        st = perf_stats(series[s], rf, bench_oos)
        st.update({"name": s, "avg_turnover": avg_turn[s]})
        leaderboard.append(st)
    bench_stats = perf_stats(bench_oos, rf)
    bench_stats["name"] = "Nifty 50 (benchmark)"
    best = max(leaderboard, key=lambda x: x["sharpe"] if x.get("sharpe") is not None else -1e9)
    for row in leaderboard:
        row["selected"] = row["name"] == best["name"]

    # --- current weights on the latest window --------------------------------
    window = returns.iloc[-LOOKBACK_DAYS:]
    cov, mu_h, mu_c, betas, shrink = estimate(window, market.loc[window.index], rf, erp)
    current = {s: weights_for(s, cov, mu_h, mu_c, rf, cap_used) for s in STRATEGIES}
    w_best = current[best["name"]]

    # Use CAPM μ for "expected" figures (historical μ is too noisy to quote)
    strategy_points = [{"name": s, "vol": float(np.sqrt(w @ cov @ w)), "ret": float(w @ mu_c),
                        "ret_hist": float(w @ mu_h)} for s, w in current.items()]
    vols = np.sqrt(np.diag(cov))
    assets = [{"ticker": t, "vol": float(vols[i]), "ret": float(mu_c[i]),
               "ret_hist": float(mu_h[i]), "beta": float(betas[i])} for i, t in enumerate(tickers)]

    port_var = w_best @ cov @ w_best
    rc = w_best * (cov @ w_best) / port_var
    div_ratio = float((w_best @ vols) / np.sqrt(port_var))

    port_daily = window @ w_best
    var_rows, p_skew, p_kurt = var_table(port_daily)
    mc = monte_carlo(window, w_best, amount)
    last_px = prices.iloc[-1]

    allocation = []
    for i, t in enumerate(tickers):
        rupees = amount * w_best[i]
        allocation.append({"ticker": t, "weight": float(w_best[i]), "risk_contribution": float(rc[i]),
                           "rupees": rupees, "price": float(last_px[t]),
                           "shares": int(rupees // last_px[t])})

    frontier = efficient_frontier(cov, mu_c, cap_used)
    corr = window.corr().round(3)

    return {
        "tickers": tickers, "errors": errors,
        "settings": {"max_weight": cap_used, "requested_max_weight": max_weight, "amount": amount,
                     "erp": erp, "rf": rf, "lookback_days": LOOKBACK_DAYS,
                     "rebalance_days": REBALANCE_DAYS, "cost_bps": COST_BPS,
                     "oos_start": str(bench_oos.index[0].date()), "oos_end": str(bench_oos.index[-1].date()),
                     "shrinkage": shrink, "mc_paths": MC_PATHS},
        "leaderboard": leaderboard, "benchmark": bench_stats, "selected": best["name"],
        "weights": {"tickers": tickers, "by_strategy": {s: w.tolist() for s, w in current.items()}},
        "strategy_points": strategy_points, "assets": assets,
        "allocation": allocation,
        "risk": {"vol": float(np.sqrt(port_var)), "expected_return_capm": float(w_best @ mu_c),
                 "expected_return_hist": float(w_best @ mu_h),
                 "beta": float(w_best @ betas), "diversification_ratio": div_ratio,
                 "skew": p_skew, "excess_kurtosis": p_kurt, "var": var_rows},
        "monte_carlo": mc["summary"],
        "correlation": {"tickers": list(corr.columns), "matrix": corr.values.tolist()},
        "frontier_chart_b64": render_frontier_chart(frontier, assets, strategy_points, rf, best["name"]),
        "backtest_chart_b64": render_backtest_chart(series, bench_oos, best["name"]),
        "mc_chart_b64": render_mc_chart(mc, amount),
        "explanations": EXPLANATIONS,
    }
