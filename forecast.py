"""
==============================================================================
forecast.py — Forecast lab (used on the Portfolio page, one stock per call)
==============================================================================
  1. Stationarity: ADF test on log price (expect unit root) and log returns
     (expect stationary) → justifies modelling returns with d = 0.
  2. Mean model: ARIMA(p,0,q) grid, p,q ∈ {0,1,2}, ranked by AIC (BIC shown).
     Honest check: the AIC winner is fitted on the training sample and scored
     1-step-ahead on the last 63 days it never saw, against two baselines
     (zero return = random walk, and the historical mean).
  3. Volatility model: GARCH(1,1)-Normal vs GARCH(1,1)-Student-t vs
     GJR-GARCH(1,1)-t (leverage effect), ranked by AIC.
  4. 21-day price fan = ARIMA drift + GARCH variance path.
==============================================================================
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from core import (
    TRADING_DAYS_PER_YEAR, ACCENT, INFO, CHART_DIM, SERIES_COLORS,
    adjusted_close, cache_get, cache_set, fetch_price_history_cached, fig_to_b64,
    legend, new_figure, set_title,
)

TEST_DAYS = 63
HORIZON = 21
ARIMA_ORDERS = list(itertools.product(range(3), range(3)))


def _fit_arima(y, order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(y, order=(order[0], 0, order[1]), trend="c").fit()


def arima_race(log_ret):
    train = log_ret.iloc[:-TEST_DAYS]
    rows = []
    for order in ARIMA_ORDERS:
        try:
            res = _fit_arima(train.values, order)
            rows.append({"order": f"ARIMA({order[0]},0,{order[1]})", "p": order[0], "q": order[1],
                         "aic": float(res.aic), "bic": float(res.bic),
                         "llf": float(res.llf), "_res": res})
        except Exception:
            continue
    rows.sort(key=lambda r: r["aic"])
    best = rows[0]
    best_bic = min(rows, key=lambda r: r["bic"])

    # 1-step-ahead out-of-sample scoring with parameters frozen on train
    applied = best["_res"].apply(log_ret.values)
    pred = np.asarray(applied.fittedvalues)[-TEST_DAYS:]
    actual = log_ret.values[-TEST_DAYS:]
    mean_fc = np.full(TEST_DAYS, train.mean())
    rmse = lambda f: float(np.sqrt(np.mean((actual - f) ** 2)))
    oos = {
        "rmse_model": rmse(pred), "rmse_random_walk": rmse(np.zeros(TEST_DAYS)),
        "rmse_hist_mean": rmse(mean_fc),
        "hit_rate": float(np.mean(np.sign(pred) == np.sign(actual))),
        "test_days": TEST_DAYS,
    }
    oos["skill_vs_rw_pct"] = (1 - oos["rmse_model"] / oos["rmse_random_walk"]) * 100
    # A sub-0.5% RMSE improvement is noise, not forecasting skill.
    oos["beats_random_walk"] = oos["skill_vs_rw_pct"] > 0.5

    full = _fit_arima(log_ret.values, (best["p"], best["q"]))
    mean_path = np.asarray(full.forecast(HORIZON))
    table = [{k: v for k, v in r.items() if k != "_res"} for r in rows]
    for r in table:
        r["best_aic"] = r["order"] == best["order"]
        r["best_bic"] = r["order"] == best_bic["order"]
    return table, best["order"], best_bic["order"], oos, mean_path, full


def garch_race(log_ret):
    y = log_ret * 100
    specs = [
        ("GARCH(1,1) · Normal", dict(vol="GARCH", p=1, q=1, dist="normal")),
        ("GARCH(1,1) · Student-t", dict(vol="GARCH", p=1, q=1, dist="t")),
        ("GJR-GARCH(1,1) · Student-t", dict(vol="GARCH", p=1, o=1, q=1, dist="t")),
    ]
    rows = []
    for name, kw in specs:
        try:
            res = arch_model(y, mean="Constant", rescale=False, **kw).fit(disp="off")
            prm = res.params
            alpha, beta = float(prm.get("alpha[1]", 0)), float(prm.get("beta[1]", 0))
            gamma = float(prm.get("gamma[1]", 0))
            persistence = alpha + beta + gamma / 2
            omega = float(prm["omega"])
            lr_var = omega / (1 - persistence) if persistence < 1 else None
            rows.append({"name": name, "aic": float(res.aic), "bic": float(res.bic),
                         "alpha": alpha, "beta": beta, "gamma": gamma if "o" in kw else None,
                         "nu": float(prm["nu"]) if "nu" in prm else None,
                         "persistence": persistence,
                         "half_life_days": float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else None,
                         "long_run_vol": float(np.sqrt(lr_var * TRADING_DAYS_PER_YEAR) / 100) if lr_var else None,
                         "_res": res})
        except Exception:
            continue
    rows.sort(key=lambda r: r["aic"])
    best = rows[0]
    res = best["_res"]
    var_path = res.forecast(horizon=HORIZON, reindex=False).variance.values[-1] / 1e4
    cond_vol = res.conditional_volatility / 100 * np.sqrt(TRADING_DAYS_PER_YEAR)
    table = [{k: v for k, v in r.items() if k != "_res"} for r in rows]
    for r in table:
        r["best_aic"] = r["name"] == best["name"]
    return table, best["name"], var_path, cond_vol


def render_fan_chart(symbol, close, mean_path, var_path):
    hist = close.iloc[-126:]
    last = float(close.iloc[-1])
    bdays = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=HORIZON)
    cum_mu, cum_sd = np.cumsum(mean_path), np.sqrt(np.cumsum(var_path))
    med = last * np.exp(cum_mu)
    fig, ax = new_figure((9, 4.2))
    ax.plot(hist.index, hist.values, color=SERIES_COLORS[0], linewidth=2, label="Price (adj.)")
    for z, alpha, lab in ((1.645, 0.15, "90% band"), (0.674, 0.30, "50% band")):
        ax.fill_between(bdays, last * np.exp(cum_mu - z * cum_sd), last * np.exp(cum_mu + z * cum_sd),
                        color=INFO, alpha=alpha, linewidth=0, label=lab)
    ax.plot(bdays, med, color=ACCENT, linewidth=2, label="Median forecast")
    set_title(ax, f"{symbol} — {HORIZON}-day forecast (ARIMA mean + GARCH variance)")
    legend(ax)
    return fig_to_b64(fig), med, cum_mu, cum_sd, last


def render_vol_chart(symbol, cond_vol, log_ret):
    realized = log_ret.rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    cv = cond_vol.iloc[-504:]
    rv = realized.reindex(cv.index)
    fig, ax = new_figure((9, 3.6))
    ax.plot(rv.index, rv.values * 100, color=CHART_DIM, linewidth=1.2, label="Realised 21-day vol")
    ax.plot(cv.index, cv.values * 100, color=SERIES_COLORS[1], linewidth=2, label="GARCH conditional vol")
    set_title(ax, f"{symbol} — volatility regime (annualised %)")
    legend(ax, ncol=2)
    return fig_to_b64(fig)


def run_forecast(symbol):
    key = f"forecast:{symbol}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    df = fetch_price_history_cached(symbol)
    if df.empty or "Close" not in df.columns:
        return {"ticker": symbol, "error": "No price data found."}
    close = adjusted_close(df)
    log_ret = np.log(close).diff().dropna()
    if len(log_ret) < 400:
        return {"ticker": symbol, "error": "Need ≥ 400 trading days of history for the forecast models."}

    adf_price = adfuller(np.log(close).values, autolag="AIC")
    adf_ret = adfuller(log_ret.values, autolag="AIC")
    jb_stat, jb_p = stats.jarque_bera(log_ret)

    arima_table, best_arima, best_arima_bic, oos, mean_path, _ = arima_race(log_ret)
    garch_table, best_garch, var_path, cond_vol = garch_race(log_ret)
    fan_b64, med, cum_mu, cum_sd, last = render_fan_chart(symbol, close, mean_path, var_path)

    points = []
    for d in (5, 10, 21):
        i = d - 1
        points.append({"days": d, "median": float(med[i]),
                       "low90": float(last * np.exp(cum_mu[i] - 1.645 * cum_sd[i])),
                       "high90": float(last * np.exp(cum_mu[i] + 1.645 * cum_sd[i])),
                       "prob_up": float(1 - stats.norm.cdf(0, cum_mu[i], cum_sd[i]))})

    out = {
        "ticker": symbol, "last_price": last, "obs": len(log_ret),
        "stationarity": {"adf_logprice_p": float(adf_price[1]), "adf_returns_p": float(adf_ret[1]),
                         "jb_p": float(jb_p), "excess_kurtosis": float(stats.kurtosis(log_ret))},
        "arima": arima_table, "best_arima": best_arima, "best_arima_bic": best_arima_bic, "oos": oos,
        "garch": garch_table, "best_garch": best_garch,
        "current_vol": float(cond_vol.iloc[-1]),
        "forecast_vol_21d": float(np.sqrt(var_path.mean() * TRADING_DAYS_PER_YEAR)),
        "points": points,
        # raw forecast paths (per trading day ahead) — used by the Smart
        # Prediction page to price any horizon from 1 to 21 days
        "mean_path": [float(x) for x in mean_path],
        "var_path": [float(x) for x in var_path],
        "garch_nu": next((g["nu"] for g in garch_table if g["best_aic"]), None),
        "last_date": str(close.index[-1].date()),
        "fan_chart_b64": fan_b64,
        "vol_chart_b64": render_vol_chart(symbol, cond_vol, log_ret),
    }
    cache_set(key, out)
    return out
