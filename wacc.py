"""
==============================================================================
wacc.py — PAGE 2 · WACC Analysis (Weighted Average Cost of Capital)
==============================================================================
Pipeline per stock
  1. Beta model race (weekly returns, same sample for every model so AIC/BIC
     are comparable):
        M1  CAPM single-index OLS            r_i = a + b·r_m
        M2  Dimson (lagged-market) beta      r_i = a + b0·r_m + b1·r_m(t-1)
        M3  Macro multi-factor MLR (APT)     r_i = a + b·r_m + USDINR + Brent + India VIX
        M4  M3 pruned by backward AIC stepwise elimination
     Every model reports coefficients (Newey-West HAC t / p), R², adj-R²,
     AIC, BIC, F-test p, Durbin-Watson, Jarque-Bera, Breusch-Pagan and VIF.
     The market-only model with the lowest AIC supplies the WACC beta, which
     is then Blume-adjusted (0.67·β + 0.33), as Bloomberg / most desks do.
  2. Cost of equity  Ke = Rf + β_adj × ERP
  3. Cost of debt    Kd = Rf + default spread from a synthetic rating
                      (interest-coverage → rating, Damodaran-style table)
  4. Weights         market cap (E) and book debt as a proxy for market debt (D)
  5. WACC            E/V·Ke + D/V·Kd·(1 − t)
  6. Two-stage FCFF DCF using that WACC + a WACC × g sensitivity grid
Banks / NBFCs / insurers are flagged: debt is their raw material, so WACC and
FCFF-DCF are not meaningful — cost of equity is the right hurdle rate.
==============================================================================
"""
import contextlib
import io

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yh
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from core import (
    ACCENT, INFO, SERIES_COLORS, CHART_DIM,
    adjusted_close, as_float, cache_get, cache_set, fetch_macro_factors,
    fetch_price_history_cached, fig_to_b64, legend, new_figure, set_title,
)

MAX_TICKERS = 6
DEFAULT_ERP = 0.0725          # India total ERP assumption (mature ERP + country risk premium)
DEFAULT_TAX_RATE = 0.2517     # India marginal corporate rate (Sec. 115BAA incl. surcharge & cess)
DEFAULT_TERMINAL_GROWTH = 0.05
BLUME_WEIGHT = 0.67
HAC_LAGS = 4
ROLLING_WEEKS = 52
FACTOR_LABELS = {"mkt": "Nifty 50", "mkt_lag1": "Nifty 50 (t-1)", "usdinr": "USD/INR",
                 "brent": "Brent crude", "india_vix": "India VIX"}

# Interest coverage → synthetic rating → default spread (large non-financial
# firms, Damodaran-style table; approximate, illustrative).
RATING_TABLE = [
    (8.50, "AAA", 0.0059), (6.50, "AA", 0.0078), (5.50, "A+", 0.0107),
    (4.25, "A", 0.0117), (3.00, "A-", 0.0128), (2.50, "BBB", 0.0162),
    (2.25, "BB+", 0.0200), (2.00, "BB", 0.0238), (1.75, "B+", 0.0314),
    (1.50, "B", 0.0330), (1.25, "B-", 0.0385), (0.80, "CCC", 0.0519),
    (0.65, "CC", 0.0851), (0.20, "C", 0.1160), (-1e9, "D", 0.1600),
]

FINANCIAL_KEYWORDS = ("bank", "insurance", "credit services", "capital markets",
                      "financial", "asset management", "mortgage")

EXPLANATIONS = {
    "beta_race": "Four regressions on the same weekly sample. Lowest AIC (Akaike) = best trade-off of fit vs. complexity; BIC penalises extra factors harder.",
    "blume": "Raw betas drift toward 1 over time. Blume adjustment (0.67·β + 0.33) is the industry-standard forward-looking beta.",
    "ke": "Cost of equity (CAPM): the return shareholders demand = risk-free rate + beta × equity risk premium.",
    "kd": "Pre-tax cost of debt: risk-free rate + a default spread implied by the firm's interest coverage (EBIT ÷ interest).",
    "wacc": "Blended hurdle rate for the whole firm. Projects (or the stock) must earn more than this to create value.",
    "dcf": "Two-stage FCFF DCF: 5 years of fading growth, then a perpetuity at the terminal growth rate, discounted at WACC.",
    "diagnostics": "DW ≈ 2 → no autocorrelation · JB p < 0.05 → non-normal residuals · BP p < 0.05 → heteroskedasticity (hence HAC errors) · VIF > 5 → multicollinearity.",
    "rolling_beta": "52-week rolling beta: shows whether market sensitivity is stable or drifting — a key check before trusting one beta number.",
}


# ==============================================================================
# REGRESSION MODELS
# ==============================================================================

def weekly_returns(series):
    return series.resample("W-FRI").last().pct_change()


def build_weekly_frame(stock_close, bench_close, factors):
    frame = pd.DataFrame({"stock": weekly_returns(stock_close),
                          "mkt": weekly_returns(bench_close)})
    frame["mkt_lag1"] = frame["mkt"].shift(1)
    for name, close in factors.items():
        frame[name] = weekly_returns(close)
    # Drop factors that would cost >15% of observations (e.g. patchy history)
    keep = ["stock", "mkt", "mkt_lag1"] + [
        f for f in factors if frame[f].notna().mean() > 0.85]
    frame = frame[keep].replace([np.inf, -np.inf], np.nan).dropna()
    return frame


def fit_ols(frame, regressors):
    X = sm.add_constant(frame[regressors])
    y = frame["stock"]
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return res, X


def summarize_model(name, spec, res, X, regressors):
    resid = res.resid
    _, bp_p, _, _ = het_breuschpagan(resid, X)
    _, jb_p, _, _ = jarque_bera(resid)
    vifs = ([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
            if len(regressors) > 1 else [1.0])
    market_beta = float(sum(res.params[r] for r in regressors if r.startswith("mkt")))
    return {
        "name": name, "spec": spec, "n": int(res.nobs), "k": len(regressors),
        "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj),
        "aic": float(res.aic), "bic": float(res.bic),
        "f_pvalue": as_float(res.f_pvalue),
        "dw": float(durbin_watson(resid)), "jb_p": float(jb_p), "bp_p": float(bp_p),
        "max_vif": float(max(vifs)),
        "market_beta": market_beta,
        "market_only": all(r.startswith("mkt") for r in regressors),
        "coefs": [{"name": "Intercept (α, weekly)" if p == "const" else FACTOR_LABELS.get(p, p),
                   "coef": float(res.params[p]), "t": float(res.tvalues[p]),
                   "p": float(res.pvalues[p])} for p in res.params.index],
    }


def backward_stepwise_aic(frame, regressors, protected=("mkt",)):
    current = list(regressors)
    best_res, best_X = fit_ols(frame, current)
    improved = True
    while improved:
        improved = False
        candidates = [r for r in current if r not in protected]
        trials = []
        for r in candidates:
            trial = [c for c in current if c != r]
            res, X = fit_ols(frame, trial)
            trials.append((res.aic, trial, res, X))
        if trials:
            aic, trial, res, X = min(trials, key=lambda t: t[0])
            if aic < best_res.aic:
                current, best_res, best_X, improved = trial, res, X, True
    return current, best_res, best_X


def run_beta_race(frame):
    factor_cols = [c for c in frame.columns if c not in ("stock", "mkt", "mkt_lag1")]
    models = []

    res, X = fit_ols(frame, ["mkt"])
    models.append(summarize_model("M1 · CAPM (single index)", "r = α + β·Nifty", res, X, ["mkt"]))

    res, X = fit_ols(frame, ["mkt", "mkt_lag1"])
    models.append(summarize_model("M2 · Dimson lagged beta", "r = α + β₀·Nifty + β₁·Nifty(t-1)",
                                  res, X, ["mkt", "mkt_lag1"]))

    stepwise_note = None
    if factor_cols:
        full = ["mkt"] + factor_cols
        res, X = fit_ols(frame, full)
        models.append(summarize_model(
            "M3 · Macro MLR (APT)",
            "r = α + β·Nifty + " + " + ".join(FACTOR_LABELS[f] for f in factor_cols),
            res, X, full))
        kept, res, X = backward_stepwise_aic(frame, full)
        if kept == ["mkt"]:
            stepwise_note = "AIC stepwise dropped every macro factor — the market alone explains this stock best."
        elif kept != full:
            models.append(summarize_model(
                "M4 · Macro MLR (AIC stepwise)",
                "r = α + β·Nifty + " + " + ".join(FACTOR_LABELS[f] for f in kept if f != "mkt"),
                res, X, kept))
        else:
            stepwise_note = "AIC stepwise kept every macro factor — M4 equals M3."

    best_aic = min(models, key=lambda m: m["aic"])
    best_bic = min(models, key=lambda m: m["bic"])
    beta_model = min((m for m in models if m["market_only"]), key=lambda m: m["aic"])
    for m in models:
        m["best_aic"] = m is best_aic
        m["best_bic"] = m is best_bic
        m["beta_source"] = m is beta_model
    return models, best_aic, best_bic, beta_model, stepwise_note


# ==============================================================================
# FUNDAMENTALS
# ==============================================================================

def _quiet(fn):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            return fn()
    except Exception:
        return None


def _row_series(df, names):
    """Return a date-ascending numeric series for the first matching row."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce").dropna()
            if len(s):
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
    return pd.Series(dtype=float)


def _latest(df, names):
    s = _row_series(df, names)
    return float(s.iloc[-1]) if len(s) else None


def fetch_fundamentals(symbol):
    key = f"fund:{symbol}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    t = yh.Ticker(symbol)
    info = _quiet(lambda: t.info) or {}
    inc = _quiet(lambda: t.income_stmt)
    bs = _quiet(lambda: t.balance_sheet)
    cf = _quiet(lambda: t.cashflow)
    out = {
        "name": info.get("longName") or info.get("shortName") or symbol,
        "sector": info.get("sector") or "", "industry": info.get("industry") or "",
        "currency": info.get("currency") or "INR",
        "fin_currency": info.get("financialCurrency") or info.get("currency") or "INR",
        "market_cap": as_float(info.get("marketCap")),
        "shares": as_float(info.get("sharesOutstanding")),
        "price": as_float(info.get("currentPrice") or info.get("regularMarketPrice")),
        "info_debt": as_float(info.get("totalDebt")),
        "info_cash": as_float(info.get("totalCash")),
        "ebit": _latest(inc, ["EBIT", "Operating Income"]),
        "interest": _latest(inc, ["Interest Expense", "Interest Expense Non Operating"]),
        "tax": _latest(inc, ["Tax Provision"]),
        "pretax": _latest(inc, ["Pretax Income"]),
        "revenue": _row_series(inc, ["Total Revenue", "Operating Revenue"]),
        "debt": _row_series(bs, ["Total Debt"]),
        "cash": _latest(bs, ["Cash Cash Equivalents And Short Term Investments",
                             "Cash And Cash Equivalents"]),
        "fcf": _row_series(cf, ["Free Cash Flow"]),
    }
    cache_set(key, out)
    return out


def synthetic_rating(coverage):
    if coverage is None:
        return "AAA", RATING_TABLE[0][2]
    for floor, rating, spread in RATING_TABLE:
        if coverage > floor:
            return rating, spread
    return "D", 0.16


def is_financial(f):
    text = f"{f['sector']} {f['industry']}".lower()
    return any(k in text for k in FINANCIAL_KEYWORDS)


# ==============================================================================
# DCF
# ==============================================================================

def two_stage_dcf(base_fcf, g1, g_term, wacc, years=5):
    fcf, pv, flows = base_fcf, 0.0, []
    for t in range(1, years + 1):
        g = g1 + (g_term - g1) * (t - 1) / (years - 1)
        fcf *= (1 + g)
        disc = fcf / (1 + wacc) ** t
        pv += disc
        flows.append({"year": t, "growth": g, "fcf": fcf, "pv": disc})
    tv = fcf * (1 + g_term) / (wacc - g_term)
    pv_tv = tv / (1 + wacc) ** years
    return pv + pv_tv, pv, pv_tv, flows


def run_dcf(f, wacc, g_term, debt, cash, shares, fx):
    fcf = f["fcf"] * fx
    if len(fcf) == 0:
        return {"ok": False, "reason": "No free-cash-flow history reported."}
    base = float(fcf.tail(3).mean())
    if base <= 0:
        return {"ok": False, "reason": "Average free cash flow (last 3 yrs) is negative — FCFF-DCF is not meaningful."}
    if not shares:
        return {"ok": False, "reason": "Shares outstanding unavailable."}
    if wacc - g_term < 0.01:
        return {"ok": False, "reason": "WACC must exceed terminal growth by ≥1 pt for a stable perpetuity."}

    rev = f["revenue"]
    if len(rev) >= 2 and rev.iloc[0] > 0:
        yrs = max(1.0, (rev.index[-1] - rev.index[0]).days / 365.25)
        hist_g = float((rev.iloc[-1] / rev.iloc[0]) ** (1 / yrs) - 1)
    else:
        hist_g = g_term
    g1 = float(np.clip(hist_g, 0.0, 0.15))

    ev, pv_stage1, pv_tv, flows = two_stage_dcf(base, g1, g_term, wacc)
    equity = ev - debt + cash
    per_share = equity / shares

    grid_w = [wacc + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
    grid_g = [g_term + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
    grid = []
    for w in grid_w:
        row = []
        for g in grid_g:
            if w - g < 0.005:
                row.append(None)
            else:
                e, *_ = two_stage_dcf(base, g1, g, w)
                row.append((e - debt + cash) / shares)
        grid.append(row)

    return {
        "ok": True, "base_fcf": base, "stage1_growth": g1, "hist_revenue_cagr": hist_g,
        "terminal_growth": g_term, "ev": ev, "equity": equity, "per_share": per_share,
        "pv_stage1": pv_stage1, "pv_terminal": pv_tv, "tv_share": pv_tv / ev,
        "flows": flows,
        "sensitivity": {"wacc": grid_w, "growth": grid_g, "values": grid},
    }


# ==============================================================================
# CHARTS
# ==============================================================================

def render_regression_chart(symbol, frame, beta_model):
    fig, ax = new_figure((9, 4.4))
    x, y = frame["mkt"] * 100, frame["stock"] * 100
    ax.scatter(x, y, s=14, color=SERIES_COLORS[0], alpha=0.55, edgecolors="none",
               label="Weekly returns")
    res = sm.OLS(frame["stock"], sm.add_constant(frame["mkt"])).fit()
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, (res.params["const"] * 100) + res.params["mkt"] * xs, color=ACCENT,
            linewidth=2, label=f"CAPM fit  β={res.params['mkt']:.2f}  R²={res.rsquared:.2f}")
    ax.axhline(0, color=CHART_DIM, linewidth=0.6, linestyle=":")
    ax.axvline(0, color=CHART_DIM, linewidth=0.6, linestyle=":")
    set_title(ax, f"{symbol} — Security Characteristic Line (weekly)")
    ax.set_xlabel("Nifty 50 weekly return (%)", fontsize=9)
    ax.set_ylabel(f"{symbol} weekly return (%)", fontsize=9)
    legend(ax)
    return fig_to_b64(fig)


def render_rolling_beta_chart(symbol, frame, full_beta):
    cov = frame["stock"].rolling(ROLLING_WEEKS).cov(frame["mkt"])
    var = frame["mkt"].rolling(ROLLING_WEEKS).var()
    rb = (cov / var).dropna()
    if rb.empty:
        return None
    fig, ax = new_figure((9, 3.6))
    ax.plot(rb.index, rb.values, color=INFO, linewidth=2, label="Rolling 52-week β")
    ax.axhline(full_beta, color=ACCENT, linestyle="--", linewidth=1.4, label=f"Full-sample β {full_beta:.2f}")
    ax.axhline(1.0, color=CHART_DIM, linestyle=":", linewidth=1, label="Market β = 1")
    set_title(ax, f"{symbol} — Beta stability")
    legend(ax, ncol=3)
    return fig_to_b64(fig)


# ==============================================================================
# ORCHESTRATION
# ==============================================================================

def analyze_wacc(symbol, ctx, erp, tax_rate, g_term):
    df = fetch_price_history_cached(symbol)
    if df.empty or "Close" not in df.columns:
        return {"ticker": symbol, "error": "No price data found for this symbol."}
    close = adjusted_close(df)

    frame = build_weekly_frame(close, ctx["bench_close"], fetch_macro_factors())
    if len(frame) < 60:
        return {"ticker": symbol, "error": f"Only {len(frame)} overlapping weeks of data — need ≥ 60 for a reliable beta."}

    models, best_aic, best_bic, beta_model, stepwise_note = run_beta_race(frame)
    raw_beta = beta_model["market_beta"]
    adj_beta = BLUME_WEIGHT * raw_beta + (1 - BLUME_WEIGHT)

    rf = ctx["rf"]
    ke = rf + adj_beta * erp
    hist_erp = ctx["market_cagr"] - rf

    f = fetch_fundamentals(symbol)
    warnings = []
    fx = 1.0
    if f["fin_currency"] != f["currency"]:
        if f["fin_currency"] == "USD" and f["currency"] == "INR":
            usdinr = fetch_macro_factors().get("usdinr")
            if usdinr is not None and len(usdinr):
                fx = float(usdinr.iloc[-1])
                warnings.append(f"Financials reported in USD — converted at ₹{fx:.2f}/$.")
        else:
            warnings.append(f"Financials reported in {f['fin_currency']}; values may be inconsistent with the {f['currency']} market cap.")

    price = float(close.iloc[-1]) if f["price"] is None else f["price"]
    market_cap = f["market_cap"] or (price * f["shares"] if f["shares"] else None)
    debt = float(f["debt"].iloc[-1]) * fx if len(f["debt"]) else (f["info_debt"] or 0.0)
    cash = (f["cash"] * fx) if f["cash"] is not None else (f["info_cash"] or 0.0)
    ebit = f["ebit"] * fx if f["ebit"] is not None else None
    interest = abs(f["interest"]) * fx if f["interest"] else None

    coverage = (ebit / interest) if (ebit is not None and interest) else None
    rating, spread = synthetic_rating(coverage)
    kd_pre = rf + spread
    observed_kd = None
    if interest and len(f["debt"]) >= 1:
        avg_debt = float(f["debt"].tail(2).mean()) * fx
        if avg_debt > 0:
            observed_kd = interest / avg_debt

    eff_tax = None
    if f["tax"] is not None and f["pretax"] and f["pretax"] > 0:
        eff_tax = f["tax"] / f["pretax"]
    kd_post = kd_pre * (1 - tax_rate)

    financial = is_financial(f)
    if not market_cap:
        return {"ticker": symbol, "error": "Market capitalisation unavailable — cannot weight capital structure."}

    V = market_cap + debt
    we, wd = market_cap / V, debt / V
    wacc = we * ke + wd * kd_post

    if financial:
        warnings.append("Financial-sector company: debt is operating raw material, so WACC / FCFF-DCF are not meaningful. Use cost of equity (Ke) as the hurdle rate.")
        dcf = {"ok": False, "reason": "Skipped for financials — value banks with P/B, ROE vs Ke or a dividend-discount model."}
    else:
        dcf = run_dcf(f, wacc, g_term, debt, cash, f["shares"], fx)
    if dcf.get("ok"):
        dcf["upside"] = dcf["per_share"] / price - 1
        if abs(dcf["upside"]) > 0.5:
            warnings.append(f"DCF value is {dcf['upside']*100:+.0f}% from the market price. The market is pricing a very different "
                            "cash-flow path (e.g. FCF depressed by a capex cycle, or growth above the 15% cap) — treat the DCF as a sanity check, not a target.")
        if dcf["tv_share"] > 0.8:
            warnings.append(f"Terminal value is {dcf['tv_share']*100:.0f}% of EV — DCF is very sensitive to the terminal assumptions.")

    return {
        "ticker": symbol, "name": f["name"], "sector": f["sector"], "industry": f["industry"],
        "is_financial": financial, "price": price,
        "weeks": len(frame), "factors_used": [FACTOR_LABELS[c] for c in frame.columns
                                              if c not in ("stock", "mkt", "mkt_lag1")],
        "models": models,
        "best_aic": best_aic["name"], "best_bic": best_bic["name"], "beta_model": beta_model["name"],
        "stepwise_note": stepwise_note,
        "raw_beta": raw_beta, "adj_beta": adj_beta,
        "rf": rf, "erp": erp, "hist_erp": hist_erp, "ke": ke,
        "coverage": coverage, "rating": rating, "spread": spread,
        "kd_pre": kd_pre, "kd_post": kd_post, "observed_kd": observed_kd,
        "tax_rate": tax_rate, "effective_tax": eff_tax,
        "market_cap": market_cap, "debt": debt, "cash": cash, "net_debt": debt - cash,
        "we": we, "wd": wd, "wacc": wacc,
        "dcf": dcf, "warnings": warnings,
        "regression_chart_b64": render_regression_chart(symbol, frame, beta_model),
        "rolling_beta_chart_b64": render_rolling_beta_chart(symbol, frame, raw_beta),
    }


def run_wacc(symbols, ctx, erp=DEFAULT_ERP, tax_rate=DEFAULT_TAX_RATE, g_term=DEFAULT_TERMINAL_GROWTH):
    results = []
    for s in symbols:
        try:
            results.append(analyze_wacc(s, ctx, erp, tax_rate, g_term))
        except Exception as exc:
            results.append({"ticker": s, "error": f"WACC analysis failed: {exc}"})
    return {
        "assumptions": {"erp": erp, "tax_rate": tax_rate, "terminal_growth": g_term,
                        "blume_weight": BLUME_WEIGHT, "hac_lags": HAC_LAGS},
        "results": results, "explanations": EXPLANATIONS,
    }
