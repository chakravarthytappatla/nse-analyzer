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
    # Fit the tails on returns divided by the running EWMA volatility known one
    # minute earlier. Dividing by one flat 5-day volatility lets volatility
    # clustering pass for fat tails (nu too low -> 1-minute ranges too narrow:
    # 86.9% inside the 90% range over 5,106 checks vs 88.7% with this fix).
    z = (r / _ewma_path(r).shift(1)).dropna().values
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
    if is_open:
        # Yahoo's daily series already holds today's unfinished bar; fitting on
        # it would treat a half-day move as a full day. While the session runs,
        # use the model as of the previous close (the one the scorecard tests).
        dm = _daily_model_before(symbol, now.normalize())
        m_path, v_path, nu_daily = np.array(dm["mean_path"]), np.array(dm["var_path"]), dm["nu"]
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


# ==============================================================================
# TODAY'S ACCURACY — how did the model do in the latest session?
# ==============================================================================
# Three honest checks, each using only data available at the time:
#   1. Opening forecasts made from the previous close (what the page showed
#      overnight): price at 09:16, at 09:25, and at today's close.
#   2. Rolling 1-minute forecasts: at every minute, predict the next minute.
#   3. Rolling 10-minute forecasts from every 10-minute mark.
# Each is scored against the real price and against the naive forecast
# ("price stays where it is").

from core import adjusted_close, disk_get, disk_set   # noqa: E402

SCORE_TTL = 60


def _ewma_path(r):
    """Per-bar EWMA minute sigma using only data up to that bar (same
    recursion and floor as minute_stats)."""
    r2 = r.values ** 2
    var = np.empty(len(r2))
    v = float(np.mean(r2[:60]))
    for i, x in enumerate(r2):
        if i >= 60:
            v = EWMA_LAMBDA * v + (1 - EWMA_LAMBDA) * x
        var[i] = v
    sig_exp = r.expanding(min_periods=30).std().bfill().values
    return pd.Series(np.maximum(np.sqrt(var), 0.5 * sig_exp), index=r.index)


def _daily_model_before(symbol, day):
    """ARIMA + GARCH fitted on daily data strictly before `day` — exactly the
    model the page would have shown at the previous close. Cached on disk."""
    key = f"openmodel-v2-{symbol}-{day.date()}"
    saved = disk_get(key, 3 * 24 * 3600)
    if saved is not None:
        return saved
    daily = fetch_price_history_cached(symbol)
    close = adjusted_close(daily)
    close = close[close.index < day.tz_localize(None)]
    log_ret = np.log(close).diff().dropna()
    _, best_arima, _, _, mean_path, _ = forecast.arima_race(log_ret)
    garch_table, best_garch, var_path, _ = forecast.garch_race(log_ret)
    nu = next((g["nu"] for g in garch_table if g["best_aic"]), None)
    prev = daily["Close"].dropna()
    prev = prev[prev.index < day.tz_localize(None)]
    out = {"mu0": float(mean_path[0]), "var0": float(var_path[0]), "nu": nu,
           "mean_path": [float(x) for x in mean_path], "var_path": [float(x) for x in var_path],
           "prev_close": float(prev.iloc[-1]), "prev_date": str(prev.index[-1].date()),
           "arima": best_arima, "garch": best_garch,
           "gap_sigma": gap_sigma(daily[daily.index < day.tz_localize(None)])}
    disk_set(key, out)
    return out


def _score(df):
    """df columns: base, pred, lo50, hi50, lo90, hi90, p_up, actual."""
    if df.empty:
        return None
    err = df["actual"] - df["pred"]
    naive = df["actual"] - df["base"]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rmse_naive = float(np.sqrt(np.mean(naive ** 2)))
    moved = naive != 0
    call_up = df["p_up"] > 0.5
    hits = (call_up & (naive > 0)) | (~call_up & (naive < 0))
    return {
        "n": int(len(df)),
        "accuracy": float(100 - np.mean(np.abs(err / df["actual"])) * 100),
        "mape": float(np.mean(np.abs(err / df["actual"])) * 100),
        "mae": float(np.mean(np.abs(err))),
        "rmse": rmse, "rmse_naive": rmse_naive,
        "theil_u": rmse / rmse_naive if rmse_naive > 0 else None,
        "hit_rate": float(hits[moved].mean() * 100) if moved.any() else None,
        "flat_pct": float((~moved).mean() * 100),
        "cover50": float(((df["actual"] >= df["lo50"]) & (df["actual"] <= df["hi50"])).mean() * 100),
        "cover90": float(((df["actual"] >= df["lo90"]) & (df["actual"] <= df["hi90"])).mean() * 100),
    }


def _bands(base, mu, sd, nu):
    q = {p: base * np.exp(quantile(p, mu, sd, nu)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
    return {"pred": q[0.5], "lo50": q[0.25], "hi50": q[0.75], "lo90": q[0.05], "hi90": q[0.95],
            "p_up": 1 - cdf(0.0, mu, sd, nu)}


def today_scorecard(symbol):
    key = f"score:{symbol}"
    cached = cache_get(key, SCORE_TTL)
    if cached is not None:
        return cached
    intra = fetch_intraday(symbol)
    if intra.empty:
        return {"ticker": symbol, "error": "No 1-minute data available right now."}
    now = now_ist()
    day = intra.index[-1].normalize()                       # latest session
    sess = intra[intra.index.normalize() == day]["Close"]
    before = intra[intra.index.normalize() < day]["Close"]
    if len(sess) < 15 or len(before) < 200:
        return {"ticker": symbol, "error": "Not enough of the session yet — check back after 09:30 IST."}
    is_live, _ = market_state(intra, now)

    # ---- per-minute sigma using only past data --------------------------------
    allc = intra["Close"]
    r = np.log(allc).groupby(allc.index.normalize()).diff().dropna()
    sig = _ewma_path(r)
    pre = r[r.index.normalize() < day]
    try:
        z = (pre / sig.shift(1).reindex(pre.index)).dropna()   # same fit as minute_stats
        nu1 = float(np.clip(stats.t.fit(z.values, floc=0)[0], 2.5, 30))
    except Exception:
        nu1 = 4.0
    dm = _daily_model_before(symbol, day)
    mu_min = dm["mu0"] / SESSION_MIN
    sig_prev = float(sig[sig.index.normalize() < day].iloc[-1])

    # ---- 1. opening forecasts from the previous close --------------------------
    p0 = dm["prev_close"]
    opening = []
    for label, minutes in (("1 minute", 1), ("10 minutes", 10)):
        target_t = day + pd.Timedelta(hours=OPEN_H, minutes=OPEN_M + minutes)
        hit = sess[sess.index >= target_t]
        if not len(hit):
            continue
        sd = float(np.sqrt(dm["gap_sigma"] ** 2 + minutes * sig_prev ** 2))
        b = _bands(p0, mu_min * minutes, sd, None)
        opening.append({"label": label, "target": hit.index[0].isoformat(), "final": True,
                        "actual": float(hit.iloc[0]), "base": p0, **b})
    b = _bands(p0, dm["mu0"], float(np.sqrt(dm["var0"])), dm["nu"])
    _, close_t = session_bounds(day)
    opening.append({"label": "1 day", "target": close_t.isoformat(),
                    "final": not is_live, "actual": float(sess.iloc[-1]),
                    "as_of": sess.index[-1].isoformat(), "base": p0, **b})
    for o in opening:
        o["error_pct"] = (o["pred"] / o["actual"] - 1) * 100
        o["in50"] = bool(o["lo50"] <= o["actual"] <= o["hi50"])
        o["in90"] = bool(o["lo90"] <= o["actual"] <= o["hi90"])
        move = o["actual"] - o["base"]
        o["direction_ok"] = None if move == 0 else bool((o["p_up"] > 0.5) == (move > 0))

    # ---- 2 & 3. rolling forecasts through the session ---------------------------
    def rolling(h, step):
        rows = []
        idx = sess.index
        for i in range(0, len(idx), step):
            t0 = idx[i]
            t1 = t0 + pd.Timedelta(minutes=h)
            if t1 not in sess.index or t0 not in sig.index and i > 0:
                continue
            s = float(sig.loc[t0]) if t0 in sig.index else sig_prev
            base = float(sess.iloc[i])
            bnd = _bands(base, mu_min * h, s * np.sqrt(h), nu1 if h == 1 else None)
            rows.append({"t0": t0, "t1": t1, "base": base, "actual": float(sess.loc[t1]), **bnd})
        return pd.DataFrame(rows)

    r1 = rolling(1, 1)
    r10 = rolling(10, 10)
    out = {
        "ticker": symbol, "session": str(day.date()), "live": bool(is_live),
        "as_of": sess.index[-1].isoformat(), "minutes_traded": int(len(sess)),
        "prev_close": p0, "prev_date": dm["prev_date"], "last_price": float(sess.iloc[-1]),
        "opening": opening,
        "rolling_1m": _score(r1), "rolling_10m": _score(r10),
        "model": {"arima": dm["arima"], "garch": dm["garch"]},
        "chart": {
            "t": [t.isoformat() for t in sess.index], "p": [round(float(x), 2) for x in sess.values],
            "f10": [{"t0": x.t0.isoformat(), "t1": x.t1.isoformat(), "pred": round(x.pred, 2),
                     "lo90": round(x.lo90, 2), "hi90": round(x.hi90, 2), "actual": round(x.actual, 2)}
                    for x in r10.itertuples()] if len(r10) else [],
        },
    }
    cache_set(key, out)
    return out
