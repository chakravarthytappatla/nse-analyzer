"""
==============================================================================
backtest.py — PAGE 04 · Model Validation (2018 → today)
==============================================================================
Question: does the Forecast-lab model (ARIMA mean picked by AIC + GARCH
variance picked by AIC) actually predict prices it has never seen?

Method — strict walk-forward, no look-ahead:
  for each test year Y in 2018 … current year:
    1. TRAIN on the 5 years of daily log returns BEFORE 1 Jan Y only
       (same 5-year window the live site uses).
    2. Re-select the ARIMA(p,0,q) order (p,q ≤ 2) and the GARCH spec by AIC
       on that training data — exactly what the live Forecast lab does.
    3. Freeze the parameters and forecast every trading day of year Y:
         * 1-day ahead  : price, 90 % interval, 95 % VaR
         * 21-day ahead : from non-overlapping monthly origins
                          (price, 90 % interval, volatility)
    4. Compare with what really happened.

Benchmarks: the naive random walk (tomorrow's price = today's price) and the
trailing 21-day realised volatility. A model only "adds value" if it beats them.

Metrics
  price   : RMSE, MAE, MAPE, RMSPE, accuracy = 100 − MAPE, Theil's U
            (= model RMSE ÷ random-walk RMSE; < 1 beats the naive forecast)
  returns : out-of-sample R² (vs. training mean), Spearman rank correlation,
            direction hit-rate
  risk    : 90 % interval coverage (should be ≈ 90 %), 95 % VaR breach rate
            (should be ≈ 5 %) with the Kupiec proportion-of-failures test
  vol     : GARCH vs naive realised-vol forecast error
  overfit : out-of-sample RMSE ÷ in-sample RMSE (≈ 1 = no overfitting)
==============================================================================
"""
import itertools
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

from core import (
    TRADING_DAYS_PER_YEAR, ACCENT, INFO, SERIES_COLORS, CHART_DIM,
    adjusted_close, cache_get, cache_set, disk_get, disk_set, fetch_price_history, fig_to_b64,
    legend, new_figure, set_title,
)

FIRST_TEST_YEAR = 2018
HISTORY_START = datetime(2012, 6, 1)   # 5-year training window for 2018 needs 2013+
TRAIN_DAYS = 5 * TRADING_DAYS_PER_YEAR
HORIZON = 21
ARIMA_ORDERS = list(itertools.product(range(3), range(3)))
GARCH_SPECS = [
    ("GARCH(1,1) · Normal", dict(vol="GARCH", p=1, q=1, dist="normal")),
    ("GARCH(1,1) · Student-t", dict(vol="GARCH", p=1, q=1, dist="t")),
    ("GJR-GARCH(1,1) · Student-t", dict(vol="GARCH", p=1, o=1, q=1, dist="t")),
]
RESULT_TTL = 6 * 60 * 60   # backtests are expensive; cache for 6 hours
YEAR_DISK_TTL = 30 * 24 * 60 * 60   # a finished year is re-scored at most monthly
CACHE_VERSION = "v1"               # bump when the method changes

EXPLANATIONS = {
    "accuracy": "Accuracy = 100 − MAPE (mean absolute % error of the predicted price). It is ALWAYS high for 1-day price forecasts because prices rarely move more than 1–2 % a day — so judge it against the naive forecast, not on its own.",
    "theil": "Theil's U = model RMSE ÷ naive random-walk RMSE. Below 1.00 the model beats 'tomorrow = today'; 1.00 means no added value; above 1.00 it is worse.",
    "coverage": "A well-calibrated 90 % interval should contain the real price about 90 % of the time. Far below = the model under-states risk; far above = bands are too wide.",
    "var": "95 % 1-day VaR should be breached on about 5 % of days. Kupiec test p > 0.05 → the breach rate is statistically consistent with 5 %.",
    "r2": "Out-of-sample R² on daily returns versus simply using the training-period average. Near 0 (or negative) is normal for stock returns — it is what market efficiency predicts.",
    "spearman": "Spearman rank correlation between predicted and actual returns: does the model at least rank good days above bad days?",
    "overfit": "Out-of-sample error ÷ in-sample error. Around 1.0 means the model performs as well on unseen data as on the data it was fitted to — no overfitting.",
    "vol": "GARCH 21-day volatility forecast vs. a naive forecast (last 21 days' realised volatility). Ratio below 1 = GARCH forecasts risk better.",
}


# ==============================================================================
# DATA
# ==============================================================================

def fetch_long_history(symbol):
    key = f"pricelong:{symbol}"
    cached = cache_get(key, RESULT_TTL)
    if cached is not None:
        return cached
    df = fetch_price_history(symbol, HISTORY_START, datetime.today())
    if not df.empty:              # never cache a failed download
        cache_set(key, df)
    return df


# ==============================================================================
# MODEL FITTING (mirrors forecast.py)
# ==============================================================================

def _fit_arima(y, order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(y, order=(order[0], 0, order[1]), trend="c").fit()


def select_arima(train):
    best = None
    for order in ARIMA_ORDERS:
        try:
            res = _fit_arima(train, order)
        except Exception:
            continue
        if best is None or res.aic < best[1].aic:
            best = (order, res)
    return best


def select_garch(y_pct, first_test_pos):
    """Fit each GARCH spec on observations before `first_test_pos` only and
    keep the AIC winner. Returns (name, fitted result, dist, nu)."""
    best = None
    for name, kw in GARCH_SPECS:
        try:
            res = arch_model(y_pct, mean="Constant", rescale=False, **kw).fit(
                disp="off", last_obs=first_test_pos)
        except Exception:
            continue
        if best is None or res.aic < best[1].aic:
            best = (name, res, kw["dist"])
    name, res, dist = best
    nu = float(res.params["nu"]) if "nu" in res.params else None
    return name, res, dist, nu


def std_quantile(p, dist, nu):
    """Quantile of the model's unit-variance innovation distribution."""
    if dist == "t" and nu and nu > 2:
        return float(stats.t.ppf(p, nu) * np.sqrt((nu - 2) / nu))
    return float(stats.norm.ppf(p))


def kupiec_pof(breaches, n, p=0.05):
    if n == 0:
        return None
    x = int(breaches)
    phat = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = ((n - x) * np.log(1 - phat) if phat < 1 else 0) + (x * np.log(phat) if x > 0 else 0)
    lr = -2 * (ll0 - ll1)
    return float(1 - stats.chi2.cdf(lr, 1))


# ==============================================================================
# ONE TEST YEAR
# ==============================================================================

def backtest_year(log_ret, price, year):
    start = pd.Timestamp(year, 1, 1)
    end = pd.Timestamp(year, 12, 31)
    test_mask = (log_ret.index >= start) & (log_ret.index <= end)
    if test_mask.sum() < 30:
        return None, None
    first_pos = int(np.argmax(test_mask))
    last_pos = int(np.where(test_mask)[0][-1])
    train_from = max(0, first_pos - TRAIN_DAYS)
    if first_pos - train_from < 2 * TRADING_DAYS_PER_YEAR:
        return None, None

    seg = log_ret.iloc[train_from:last_pos + 1]          # train + this test year
    n_train = first_pos - train_from
    train = seg.iloc[:n_train]
    test = seg.iloc[n_train:]

    # ---- mean model: ARIMA chosen by AIC on TRAIN only --------------------
    order, ares = select_arima(train.values)
    applied = ares.apply(seg.values)                   # params frozen
    mu_hat = np.asarray(applied.fittedvalues)[n_train:]  # 1-step predictions
    in_sample_rmse = float(np.sqrt(np.mean(np.asarray(ares.resid) ** 2)))

    # ---- variance model: GARCH chosen by AIC on TRAIN only ----------------
    gname, gres, dist, nu = select_garch(seg * 100, n_train)
    fc = gres.forecast(horizon=HORIZON, start=n_train - 1, reindex=False)
    var_paths = fc.variance.values / 1e4                # row k = origin seg[n_train-1+k]
    sigma1 = np.sqrt(var_paths[:len(test), 0])           # 1-day sigma for each test day

    # ---- 1-day price forecasts -------------------------------------------
    prev_px = price.reindex(seg.index).shift(1).iloc[n_train:].values
    actual_px = price.reindex(test.index).values
    pred_px = prev_px * np.exp(mu_hat)
    q05, q95 = std_quantile(0.05, dist, nu), std_quantile(0.95, dist, nu)
    lo90 = prev_px * np.exp(mu_hat + q05 * sigma1)
    hi90 = prev_px * np.exp(mu_hat + q95 * sigma1)
    var95 = -(mu_hat + q05 * sigma1)                     # as positive loss (log-return)
    actual_r = test.values

    err = actual_px - pred_px
    rw_err = actual_px - prev_px
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rw_rmse = float(np.sqrt(np.mean(rw_err ** 2)))
    ape = np.abs(err / actual_px)
    oos_rmse_r = float(np.sqrt(np.mean((actual_r - mu_hat) ** 2)))
    sst = np.sum((actual_r - train.mean()) ** 2)
    r2_os = float(1 - np.sum((actual_r - mu_hat) ** 2) / sst) if sst > 0 else None
    rho = stats.spearmanr(mu_hat, actual_r).correlation if np.std(mu_hat) > 0 else None
    breaches = int(np.sum(-actual_r > var95))

    # ---- 21-day forecasts from non-overlapping monthly origins -----------
    rows21 = []
    for k in range(0, len(test) - HORIZON + 1, HORIZON):
        origin_pos = n_train - 1 + k                      # last known close
        hist = seg.values[:origin_pos + 1]
        mean_path = np.asarray(ares.apply(hist).forecast(HORIZON))
        vpath = var_paths[k]                              # variance path from that origin
        cum_mu, cum_sd = mean_path.sum(), np.sqrt(vpath.sum())
        p0 = float(price.loc[seg.index[origin_pos]])
        target_date = test.index[k + HORIZON - 1]
        p_true = float(price.loc[target_date])
        realized_vol = float(np.std(test.values[k:k + HORIZON], ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        naive_vol = float(np.std(seg.values[origin_pos - HORIZON + 1:origin_pos + 1], ddof=1)
                          * np.sqrt(TRADING_DAYS_PER_YEAR))
        rows21.append({
            "origin": seg.index[origin_pos], "date": target_date, "p0": p0, "actual": p_true,
            "pred": p0 * np.exp(cum_mu),
            "lo90": p0 * np.exp(cum_mu - 1.645 * cum_sd), "hi90": p0 * np.exp(cum_mu + 1.645 * cum_sd),
            "pred_vol": float(np.sqrt(vpath.mean() * TRADING_DAYS_PER_YEAR)),
            "naive_vol": naive_vol, "realized_vol": realized_vol,
        })
    d21 = pd.DataFrame(rows21)

    out = {
        "year": year, "days": int(len(test)),
        "test_from": str(test.index[0].date()), "test_to": str(test.index[-1].date()),
        "train_from": str(train.index[0].date()), "train_to": str(train.index[-1].date()),
        "arima": f"ARIMA({order[0]},0,{order[1]})", "garch": gname,
        # 1-day price
        "rmse": rmse, "mae": float(np.mean(np.abs(err))), "mape": float(ape.mean() * 100),
        "rmspe": float(np.sqrt(np.mean(ape ** 2)) * 100),
        "accuracy": float(100 - ape.mean() * 100),
        "rw_rmse": rw_rmse, "rw_mape": float(np.mean(np.abs(rw_err / actual_px)) * 100),
        "theil_u": rmse / rw_rmse if rw_rmse > 0 else None,
        # returns
        "r2_os": r2_os, "spearman": None if rho is None or np.isnan(rho) else float(rho),
        "hit_rate": float(np.mean(np.sign(mu_hat) == np.sign(actual_r)) * 100),
        "up_days": float(np.mean(actual_r > 0) * 100),
        "overfit_ratio": oos_rmse_r / in_sample_rmse if in_sample_rmse > 0 else None,
        # risk
        "coverage90": float(np.mean((actual_px >= lo90) & (actual_px <= hi90)) * 100),
        "var_breach": float(breaches / len(test) * 100), "var_breaches": breaches,
        "kupiec_p": kupiec_pof(breaches, len(test)),
    }
    if len(d21):
        e21 = d21["actual"] - d21["pred"]
        rw21 = d21["actual"] - d21["p0"]
        ve = d21["realized_vol"] - d21["pred_vol"]
        vn = d21["realized_vol"] - d21["naive_vol"]
        out.update({
            "n21": int(len(d21)),
            "mape21": float(np.mean(np.abs(e21 / d21["actual"])) * 100),
            "accuracy21": float(100 - np.mean(np.abs(e21 / d21["actual"])) * 100),
            "rmse21": float(np.sqrt(np.mean(e21 ** 2))),
            "theil_u21": float(np.sqrt(np.mean(e21 ** 2)) / np.sqrt(np.mean(rw21 ** 2)))
                         if np.mean(rw21 ** 2) > 0 else None,
            "hit21": float(np.mean(np.sign(d21["pred"] - d21["p0"]) == np.sign(rw21)) * 100),
            "coverage21": float(np.mean((d21["actual"] >= d21["lo90"]) & (d21["actual"] <= d21["hi90"])) * 100),
            "vol_mape": float(np.mean(np.abs(ve / d21["realized_vol"])) * 100),
            "vol_ratio": float(np.sqrt(np.mean(ve ** 2)) / np.sqrt(np.mean(vn ** 2)))
                         if np.mean(vn ** 2) > 0 else None,
        })
    daily = pd.DataFrame({"actual": actual_px, "pred": pred_px, "rw": prev_px,
                          "lo90": lo90, "hi90": hi90,
                          "actual_r": actual_r, "mu": mu_hat, "var95": var95}, index=test.index)
    return out, {"daily": daily, "d21": d21}


# ==============================================================================
# POOLED SUMMARY + CHARTS
# ==============================================================================

def pooled(rows, details):
    daily = pd.concat([d["daily"] for d in details])
    d21 = pd.concat([d["d21"] for d in details if len(d["d21"])])
    err = daily["actual"] - daily["pred"]
    rw = daily["actual"] - daily["rw"]
    ape = np.abs(err / daily["actual"])
    e21 = d21["actual"] - d21["pred"]
    rw21 = d21["actual"] - d21["p0"]
    breaches = int(np.sum(-daily["actual_r"] > daily["var95"]))
    ve = d21["realized_vol"] - d21["pred_vol"]
    vn = d21["realized_vol"] - d21["naive_vol"]
    return {
        "days": int(len(daily)), "n21": int(len(d21)),
        "accuracy": float(100 - ape.mean() * 100), "mape": float(ape.mean() * 100),
        "rmspe": float(np.sqrt(np.mean(ape ** 2)) * 100),
        "theil_u": float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(rw ** 2))),
        "hit_rate": float(np.mean(np.sign(daily["mu"]) == np.sign(daily["actual_r"])) * 100),
        "spearman": float(stats.spearmanr(daily["mu"], daily["actual_r"]).correlation)
                    if daily["mu"].std() > 0 else None,
        "coverage90": float(np.mean([r["coverage90"] for r in rows])),
        "var_breach": breaches / len(daily) * 100, "kupiec_p": kupiec_pof(breaches, len(daily)),
        "accuracy21": float(100 - np.mean(np.abs(e21 / d21["actual"])) * 100),
        "theil_u21": float(np.sqrt(np.mean(e21 ** 2)) / np.sqrt(np.mean(rw21 ** 2))),
        "hit21": float(np.mean(np.sign(d21["pred"] - d21["p0"]) == np.sign(rw21)) * 100),
        "coverage21": float(np.mean((d21["actual"] >= d21["lo90"]) & (d21["actual"] <= d21["hi90"])) * 100),
        "vol_ratio": float(np.sqrt(np.mean(ve ** 2)) / np.sqrt(np.mean(vn ** 2))),
        "overfit_ratio": float(np.mean([r["overfit_ratio"] for r in rows if r["overfit_ratio"]])),
    }


def render_price_chart(symbol, price, d21):
    fig, ax = new_figure((10, 4.6))
    px = price[price.index >= pd.Timestamp(FIRST_TEST_YEAR, 1, 1)]
    ax.plot(px.index, px.values, color=SERIES_COLORS[0], linewidth=1.4, label="Actual price")
    yerr = np.vstack([d21["pred"] - d21["lo90"], d21["hi90"] - d21["pred"]])
    ax.errorbar(d21["date"], d21["pred"], yerr=yerr, fmt="o", color=ACCENT, ecolor=ACCENT,
                elinewidth=1, capsize=0, markersize=4, alpha=0.9,
                label="21-day forecast (median + 90% range)")
    for y in range(FIRST_TEST_YEAR + 1, datetime.today().year + 1):
        ax.axvline(pd.Timestamp(y, 1, 1), color=CHART_DIM, linewidth=0.6, linestyle=":")
    set_title(ax, f"{symbol} — walk-forward forecasts vs actual price, {FIRST_TEST_YEAR}–today")
    ax.set_ylabel("Price (₹, adjusted)", fontsize=9)
    legend(ax)
    return fig_to_b64(fig)


def render_vol_chart(symbol, d21):
    fig, ax = new_figure((10, 3.6))
    ax.plot(d21["date"], d21["realized_vol"] * 100, color=CHART_DIM, linewidth=1.3,
            label="Realised 21-day volatility")
    ax.plot(d21["date"], d21["pred_vol"] * 100, color=SERIES_COLORS[1], linewidth=2,
            label="GARCH forecast (made 21 days earlier)")
    set_title(ax, f"{symbol} — predicted vs realised volatility (annualised %)")
    legend(ax, ncol=2)
    return fig_to_b64(fig)


def run_backtest(symbol):
    key = f"backtest:{symbol}"
    cached = cache_get(key, RESULT_TTL)
    if cached is not None:
        return cached
    df = fetch_long_history(symbol)
    if df.empty or "Close" not in df.columns:
        return {"ticker": symbol, "error": "No price data found."}
    price = adjusted_close(df)
    log_ret = np.log(price).diff().dropna()

    rows, details = [], []
    this_year = datetime.today().year
    for year in range(FIRST_TEST_YEAR, this_year + 1):
        # Completed years never change, so they are scored once and kept on
        # disk; only the current (still running) year is recomputed.
        disk_key = f"bt-{CACHE_VERSION}-{symbol}-{year}"
        saved = disk_get(disk_key, YEAR_DISK_TTL) if year < this_year else None
        if saved is not None:
            row, detail = saved
        else:
            try:
                row, detail = backtest_year(log_ret, price, year)
                if year < this_year and row is not None:
                    disk_set(disk_key, (row, detail))
            except Exception as exc:
                row, detail = {"year": year, "error": str(exc)}, None
        if row is None:
            continue
        rows.append(row)
        if detail is not None:
            details.append(detail)
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"ticker": symbol, "error": "Not enough history: needs ≥ 2 years before 2018 for a fair test."}

    d21_all = pd.concat([d["d21"] for d in details if len(d["d21"])])
    daily_all = pd.concat([d["daily"] for d in details])
    out = {
        "ticker": symbol, "first_year": ok[0]["year"], "years": rows,
        "summary": pooled(ok, details),
        "price_chart_b64": render_price_chart(symbol, price, d21_all),
        "vol_chart_b64": render_vol_chart(symbol, d21_all),
        # every real-vs-predicted value, so the user can audit it (table + CSV)
        "forecasts21": [{"origin": str(r.origin.date()), "date": str(r.date.date()), "p0": r.p0,
                         "pred": r.pred, "lo90": r.lo90, "hi90": r.hi90, "actual": r.actual,
                         "pred_vol": r.pred_vol, "realized_vol": r.realized_vol}
                        for r in d21_all.itertuples()],
        "daily": {"date": [str(d.date()) for d in daily_all.index],
                  "actual": daily_all["actual"].round(2).tolist(),
                  "pred": daily_all["pred"].round(2).tolist(),
                  "naive": daily_all["rw"].round(2).tolist(),
                  "lo90": daily_all["lo90"].round(2).tolist(),
                  "hi90": daily_all["hi90"].round(2).tolist()},
        "explanations": EXPLANATIONS,
    }
    cache_set(key, out)
    return out
