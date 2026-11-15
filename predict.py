"""
==============================================================================
predict.py — PAGE 04 · Smart Prediction
==============================================================================
For one NSE stock, a probability forecast of the price at six horizons:

    1 minute · 10 minutes · 1 day · 2 days · 1 week (5 sessions) · 1 month (21)

Each horizon is a distribution of the log return  r ~ (μ, σ, ν):
  * minutes  — μ = daily ARIMA drift ÷ 375 per minute; σ from an EWMA
               (λ = 0.97) of today's/recent 1-minute returns, floored at half
               the 5-day minute volatility; Student-t tails for 1 minute.
               If the market is shut, the overnight opening-gap variance
               (from 250 days of open ÷ previous close) is added, because the
               "next minute" is then the minute after the next open.
  * days     — the Forecast-lab model (ARIMA mean picked by AIC + GARCH
               variance picked by AIC, the model validated walk-forward
               2018-2026). During market hours only the unexpired fraction of
               today's session is left for the 1-day horizon.
Outputs per horizon: median price, 50 % / 90 % ranges, P(up), P(down), and the
(μ, σ, ν) so the browser can price any target interactively.

NSE session: 09:15–15:30 IST (375 minutes), Monday–Friday. Exchange holidays
are not known to the model, so a target date may land on a holiday.
==============================================================================
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

import forecast
from core import (
    cache_get, cache_set, fetch_intraday, fetch_price_history_cached,
)

IST = "Asia/Kolkata"
OPEN_H, OPEN_M, CLOSE_H, CLOSE_M = 9, 15, 15, 30
SESSION_MIN = 375
EWMA_LAMBDA = 0.97
GAP_DAYS = 250
STALE_MINUTES = 20          # last bar older than this during hours → treat as closed
HORIZONS = [
    {"key": "1m", "label": "1 minute", "unit": "min", "n": 1},
    {"key": "10m", "label": "10 minutes", "unit": "min", "n": 10},
    {"key": "1d", "label": "1 day", "unit": "day", "n": 1},
    {"key": "2d", "label": "2 days", "unit": "day", "n": 2},
    {"key": "1w", "label": "1 week", "unit": "day", "n": 5},
    {"key": "1mo", "label": "1 month", "unit": "day", "n": 21},
]


# ==============================================================================
# CALENDAR
# ==============================================================================

def now_ist():
    return pd.Timestamp.now(tz=IST)


def session_bounds(day):
    d = pd.Timestamp(day).tz_convert(IST) if pd.Timestamp(day).tzinfo else pd.Timestamp(day).tz_localize(IST)
    d = d.normalize()
    return d + pd.Timedelta(hours=OPEN_H, minutes=OPEN_M), d + pd.Timedelta(hours=CLOSE_H, minutes=CLOSE_M)


def is_weekday(ts):
    return ts.weekday() < 5


def next_weekday(day):
    d = day + pd.Timedelta(days=1)
    while not is_weekday(d):
        d += pd.Timedelta(days=1)
    return d


def next_open(now):
    o, _ = session_bounds(now)
    if is_weekday(now) and now < o:
        return o
    return session_bounds(next_weekday(now.normalize()))[0]


def add_trading_days(day, k):
    d = pd.Timestamp(day).normalize()
    for _ in range(k):
        d = next_weekday(d)
    return d


# ==============================================================================
# INTRADAY STATISTICS
# ==============================================================================

def minute_stats(intra):
    closes = intra["Close"]
    day = closes.index.normalize()
    r = np.log(closes).groupby(day).diff().dropna()      # within-session only
    r = r[np.isfinite(r)]
    if len(r) < 60:
        return None
    r2 = r.values ** 2
    var = float(np.mean(r2[:60]))
    for x in r2[60:]:
        var = EWMA_LAMBDA * var + (1 - EWMA_LAMBDA) * x
    sigma_ewma = float(np.sqrt(var))
    sigma_5d = float(r.std())
    sigma = max(sigma_ewma, 0.5 * sigma_5d)
    z = r.values / sigma_5d
    try:
        nu = float(stats.t.fit(z, floc=0)[0])
    except Exception:
        nu = 4.0
    return {"sigma": sigma, "sigma_ewma": sigma_ewma, "sigma_5d": sigma_5d,
            "nu": float(np.clip(nu, 2.5, 30)), "n": int(len(r))}


def gap_sigma(daily):
    d = daily.dropna(subset=["Open", "Close"]).tail(GAP_DAYS + 1)
    g = np.log(d["Open"] / d["Close"].shift(1)).dropna()
    lo, hi = np.percentile(g, [1, 99])
    return float(g.clip(lo, hi).std())


# ==============================================================================
# DISTRIBUTION HELPERS
# ==============================================================================

def quantile(p, mu, sd, nu):
    if nu:
        return mu + sd * stats.t.ppf(p, nu) * np.sqrt((nu - 2) / nu)
    return mu + sd * stats.norm.ppf(p)


def cdf(x, mu, sd, nu):
    if sd <= 0:
        return float(x >= mu)
    z = (x - mu) / sd
    if nu:
        return float(stats.t.cdf(z / np.sqrt((nu - 2) / nu), nu))
    return float(stats.norm.cdf(z))


def summarize(h, base, mu, sd, nu, target_time, note=None):
    q = {p: base * np.exp(quantile(p, mu, sd, nu)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
    p_up = 1 - cdf(0.0, mu, sd, nu)
    return {**h, "base": base, "mu": mu, "sd": sd, "nu": nu,
            "target_time": target_time.isoformat(), "median": q[0.5],
            "lo50": q[0.25], "hi50": q[0.75], "lo90": q[0.05], "hi90": q[0.95],
            "change_pct": (q[0.5] / base - 1) * 100, "typical_move_pct": sd * 100,
            "p_up": p_up, "p_down": 1 - p_up, "note": note}


# ==============================================================================
# ORCHESTRATION
# ==============================================================================

def market_state(intra, now):
    o, c = session_bounds(now)
    in_hours = is_weekday(now) and o <= now < c
    last_bar = intra.index[-1] if len(intra) else None
    fresh = last_bar is not None and (now - last_bar) <= pd.Timedelta(minutes=STALE_MINUTES) \
        and last_bar.normalize() == now.normalize()
    return in_hours and fresh, last_bar


def run_predict(symbol):
    key = f"predict:{symbol}"
    cached = cache_get(key, 60)
    if cached is not None:
        return cached

    daily = fetch_price_history_cached(symbol)
    if daily.empty or "Close" not in daily.columns:
        return {"ticker": symbol, "error": "No price data found for this symbol."}
    fc = forecast.run_forecast(symbol)
    if fc.get("error"):
        return {"ticker": symbol, "error": fc["error"]}
    intra = fetch_intraday(symbol)

    now = now_ist()
    is_open, last_bar = market_state(intra, now) if not intra.empty else (False, None)
    m_path, v_path = np.array(fc["mean_path"]), np.array(fc["var_path"])
    nu_daily = fc.get("garch_nu")
    last_close = float(daily["Close"].dropna().iloc[-1])
    last_close_date = pd.Timestamp(daily["Close"].dropna().index[-1]).tz_localize(IST)
    live_price = float(intra["Close"].iloc[-1]) if is_open else last_close
    ms = minute_stats(intra) if not intra.empty else None
    gsig = gap_sigma(daily)
    mu_min = m_path[0] / SESSION_MIN

    out_h = []
    # ---------------- minute horizons ----------------------------------------
    for h in (x for x in HORIZONS if x["unit"] == "min"):
        n = h["n"]
        if ms is None:
            out_h.append({**h, "error": "Minute data unavailable right now."})
            continue
        var, note = n * ms["sigma"] ** 2, None
        if is_open:
            _, close_t = session_bounds(now)
            start = last_bar
            mins_left = int((close_t - start).total_seconds() // 60)
            if n <= mins_left:
                target = start + pd.Timedelta(minutes=n)
            else:
                var += gsig ** 2
                target = next_open(close_t) + pd.Timedelta(minutes=n - mins_left)
                note = "Crosses the close — includes the overnight gap risk."
        else:
            var += gsig ** 2
            target = next_open(now) + pd.Timedelta(minutes=n)
            note = "Market closed — measured from the last close, so it includes the opening-gap risk."
        out_h.append(summarize(h, live_price, mu_min * n, float(np.sqrt(var)),
                               ms["nu"] if n == 1 and not note else None, target, note))

    # ---------------- day horizons -------------------------------------------
    if is_open:
        _, close_t = session_bounds(now)
        frac = max(0.0, min(1.0, (close_t - last_bar).total_seconds() / 60 / SESSION_MIN))
        first_day = now.normalize()
    else:
        frac = 1.0
        first_day = add_trading_days(last_close_date, 1)
    for h in (x for x in HORIZONS if x["unit"] == "day"):
        n = h["n"]
        mu = frac * m_path[0] + m_path[1:n].sum()
        var = frac * v_path[0] + v_path[1:n].sum()
        target = session_bounds(add_trading_days(first_day, n - 1))[1]
        out_h.append(summarize(h, live_price, float(mu), float(np.sqrt(var)),
                               nu_daily if n == 1 else None, target))

    # ---------------- forecast cones for the chart ----------------------------
    day_path = []
    for k in range(1, len(m_path) + 1):
        mu_k = frac * m_path[0] + m_path[1:k].sum()
        sd_k = float(np.sqrt(frac * v_path[0] + v_path[1:k].sum()))
        day_path.append({"t": session_bounds(add_trading_days(first_day, k - 1))[1].isoformat(),
                         "mu": float(mu_k), "sd": sd_k})
    minute_path = []
    if ms is not None:
        start = last_bar if is_open else next_open(now)
        gap_var = 0.0 if is_open else gsig ** 2
        for k in range(1, 11):
            minute_path.append({"t": (start + pd.Timedelta(minutes=k)).isoformat(),
                                "mu": float(mu_min * k),
                                "sd": float(np.sqrt(gap_var + k * ms["sigma"] ** 2))})

    # ---------------- chart history ------------------------------------------
    dclose = daily["Close"].dropna().tail(90)
    hist_daily = {"t": [str(d.date()) for d in dclose.index], "p": [round(float(x), 2) for x in dclose.values]}
    hist_intra = {"t": [], "p": []}
    if not intra.empty:
        last_session = intra[intra.index.normalize() == intra.index[-1].normalize()]["Close"]
        hist_intra = {"t": [t.isoformat() for t in last_session.index],
                      "p": [round(float(x), 2) for x in last_session.values]}

    nxt = next_open(now)
    out = {
        "ticker": symbol, "price": live_price, "last_close": last_close,
        "last_close_date": str(last_close_date.date()),
        "market_open": bool(is_open), "now": now.isoformat(),
        "last_tick": last_bar.isoformat() if last_bar is not None else None,
        "next_open": nxt.isoformat(),
        "horizons": out_h,
        "paths": {"day": day_path, "minute": minute_path},
        "history": {"daily": hist_daily, "intraday": hist_intra},
        "model": {
            "daily_mean": fc["best_arima"], "daily_vol": fc["best_garch"],
            "garch_nu": nu_daily,
            "minute_sigma_pct": ms["sigma"] * 100 if ms else None,
            "minute_nu": ms["nu"] if ms else None,
            "minute_obs": ms["n"] if ms else 0,
            "gap_sigma_pct": gsig * 100,
            "daily_vol_annual_pct": fc["forecast_vol_21d"] * 100,
        },
    }
    cache_set(key, out)
    return out
