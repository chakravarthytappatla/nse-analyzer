"""
==============================================================================
core.py — shared plumbing for all three pages
==============================================================================
  * config constants
  * a tiny in-process TTL cache
  * price / macro data acquisition (Yahoo Finance + FRED)
  * chart helpers (thread-safe matplotlib Figure API, dark terminal theme)
  * JSON sanitising (NaN / inf -> null so the browser can always parse it)
==============================================================================
"""
import base64
import contextlib
import io
import logging
import math
import os
import threading
import time
import warnings
from datetime import datetime, timedelta

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # OO API: safe under gunicorn threads
import numpy as np
import pandas as pd
import requests
import yfinance as yh

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ==============================================================================
# CONFIG
# ==============================================================================

BENCHMARK_TICKER = "^NSEI"

# India 10Y G-Sec. Yahoo no longer serves any India 10Y symbol, so the primary
# source is FRED's OECD series (monthly, no API key). Yahoo candidates are kept
# as a secondary attempt in case they come back.
FRED_INDIA_10Y_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=INDIRLTLT01STM"
BOND_YIELD_TICKER_CANDIDATES = ["^IN10YT=X", "IN10Y=X", "^IN10Y"]
FALLBACK_RISK_FREE_RATE = 0.070

# Macro factors used by the multi-factor (MLR) regression on the WACC page.
MACRO_FACTOR_TICKERS = {
    "usdinr": "INR=X",       # rupee weakness -> helps exporters (IT, pharma)
    "brent": "BZ=F",         # crude oil -> input cost for OMCs, paints, airlines
    "india_vix": "^INDIAVIX",  # fear gauge -> risk-off sensitivity
}

TRADING_DAYS_PER_YEAR = 252
HISTORY_YEARS = 5
CACHE_TTL_SECONDS = 15 * 60

# Categorical palette (dark-surface steps, fixed order — colour follows the
# entity, never its rank). Status colours (gain/loss) are kept separate.
SERIES_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500",
                 "#d55181", "#008300", "#9085e9", "#e66767"]
CHART_BG = "#12161c"
CHART_TEXT = "#f2f2f2"
CHART_DIM = "#9aa4b2"
CHART_GRID = "#5a6577"
CHART_SPINE = "#2a313c"
ACCENT = "#ffb347"
INFO = "#5ecbf1"
GAIN = "#37d67a"
LOSS = "#ff5c5c"

_cache = {}


def cache_get(key, ttl=CACHE_TTL_SECONDS):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < ttl:
        return entry["value"]
    return None


def cache_set(key, value):
    _cache[key] = {"value": value, "ts": time.time()}


# ------------------------------------------------------------------------------
# Persistent disk cache — survives restarts. On Azure App Service /home is a
# persistent share; locally it falls back to the system temp folder.
# ------------------------------------------------------------------------------
import pickle
import tempfile

DATA_DIR = os.environ.get("NSE_DATA_DIR") or (
    "/home/data/nse-terminal" if os.path.isdir("/home/site") else
    os.path.join(tempfile.gettempdir(), "nse-terminal-cache"))


def _disk_path(key):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return os.path.join(DATA_DIR, safe + ".pkl")


def disk_get(key, max_age_seconds):
    path = _disk_path(key)
    try:
        if time.time() - os.path.getmtime(path) > max_age_seconds:
            return None
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ImportError):
        return None


def disk_set(key, value):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = _disk_path(key) + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, _disk_path(key))   # atomic: readers never see half a file
    except OSError:
        pass                               # cache is best-effort


def date_range(years=HISTORY_YEARS):
    end = datetime.today()
    start = end - timedelta(days=int(years * 365))
    return start, end


def normalize_symbols(raw_symbols, limit):
    symbols = []
    for s in raw_symbols or []:
        s = str(s or "").strip().upper()
        if not s:
            continue
        if not s.endswith(".NS"):
            s = f"{s}.NS"
        if s not in symbols:
            symbols.append(s)
    return symbols[:limit]


def as_float(value, default=None):
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def clean_json(obj):
    """Recursively replace NaN/inf with None and numpy scalars with Python
    types. Flask would otherwise emit bare `NaN`, which is invalid JSON and
    makes `response.json()` throw in the browser."""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ==============================================================================
# DATA ACQUISITION
# ==============================================================================

# yfinance keeps download results in a module-level dict and we silence its
# prints by swapping the process-wide stdout/stderr — neither is thread-safe.
# Under gunicorn threads two concurrent downloads can return an empty or even
# the wrong ticker's frame, so every Yahoo call goes through this one lock.
YAHOO_LOCK = threading.RLock()
DOWNLOAD_RETRIES = 3


def _download_once(ticker, start, end):
    buf = io.StringIO()
    with YAHOO_LOCK, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        df = yh.download(ticker, start=start, end=end, progress=False,
                         auto_adjust=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        # guard against a frame that belongs to another ticker
        tickers = set(df.columns.get_level_values(-1))
        if tickers and ticker not in tickers:
            return pd.DataFrame()
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna(how="all")


def fetch_price_history(ticker, start, end):
    """OHLCV for one ticker. Retries transient empty responses from Yahoo."""
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            df = _download_once(ticker, start, end)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            return df
        if attempt < DOWNLOAD_RETRIES - 1:
            time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame()


def fetch_price_history_cached(ticker):
    key = f"price:{ticker}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    start, end = date_range()
    df = fetch_price_history(ticker, start, end)
    if not df.empty:              # never cache a failed download
        cache_set(key, df)
    return df


INTRADAY_TTL_SECONDS = 60


def fetch_intraday(ticker):
    """Last ~5 sessions of 1-minute bars (index in Asia/Kolkata). Cached 60 s."""
    key = f"intraday:{ticker}"
    cached = cache_get(key, INTRADAY_TTL_SECONDS)
    if cached is not None:
        return cached
    df = pd.DataFrame()
    for attempt in range(DOWNLOAD_RETRIES):
        buf = io.StringIO()
        try:
            with YAHOO_LOCK, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                df = yh.download(ticker, period="5d", interval="1m", progress=False,
                                 auto_adjust=False, threads=False)
        except Exception:
            df = pd.DataFrame()
        if df is not None and not df.empty:
            break
        time.sleep(1.5 * (attempt + 1))
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if ticker not in set(df.columns.get_level_values(-1)):
            return pd.DataFrame()
        df.columns = df.columns.get_level_values(0)
    idx = pd.to_datetime(df.index)
    df.index = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert("Asia/Kolkata")
    df = df.dropna(subset=["Close"])
    cache_set(key, df)
    return df


def adjusted_close(df):
    """Total-return series (dividends & splits adjusted) when available."""
    if "Adj Close" in df.columns and df["Adj Close"].notna().sum() > 0:
        return df["Adj Close"].dropna()
    return df["Close"].dropna()


def fetch_benchmark():
    df = fetch_price_history_cached(BENCHMARK_TICKER)
    if df.empty or "Close" not in df.columns:
        return None
    return df


def _fetch_fred_india_10y():
    # NB: FRED rejects a bare "Mozilla/5.0" UA as a bot; the default requests UA works.
    resp = requests.get(FRED_INDIA_10Y_URL, timeout=10)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().set_index("date")["value"]
    start, _ = date_range()
    return s[s.index >= pd.Timestamp(start) - pd.Timedelta(days=31)]


def fetch_risk_free():
    """Returns dict: rate (decimal), yields (pd.Series of % values), source,
    as_of (ISO date). Order: FRED -> Yahoo -> fixed fallback."""
    key = "risk_free"
    cached = cache_get(key)
    if cached is not None:
        return cached

    result = None
    try:
        s = _fetch_fred_india_10y()
        if len(s) >= 12:
            result = {"rate": float(s.iloc[-1]) / 100, "yields": s,
                      "source": "FRED · OECD India 10Y G-Sec (monthly)",
                      "as_of": s.index[-1].strftime("%Y-%m-%d")}
    except Exception:
        result = None

    if result is None:
        start, end = date_range()
        for candidate in BOND_YIELD_TICKER_CANDIDATES:
            df = fetch_price_history(candidate, start, end)
            if not df.empty and "Close" in df.columns and df["Close"].notna().sum() > 30:
                s = df["Close"].dropna()
                result = {"rate": float(s.iloc[-1]) / 100, "yields": s,
                          "source": f"Yahoo {candidate}",
                          "as_of": s.index[-1].strftime("%Y-%m-%d")}
                break

    if result is None:
        # not cached: the next request retries the live sources
        return {"rate": FALLBACK_RISK_FREE_RATE, "yields": pd.Series(dtype=float),
                "source": "Fallback assumption (live yield unavailable)",
                "as_of": None}

    cache_set(key, result)
    return result


def macro_yield_commentary(yields):
    """Stance on rates using a time-based window, so it works for monthly
    (FRED) and daily (Yahoo) series alike."""
    yields = yields.dropna() if yields is not None else pd.Series(dtype=float)
    if len(yields) < 12:
        return "NEUTRAL", ("Insufficient bond-yield history is available right now to "
                           "assess the macro backdrop, so a neutral stance is assumed.")

    start_yield, end_yield = float(yields.iloc[0]), float(yields.iloc[-1])
    change_bps = (end_yield - start_yield) * 100
    six_months_ago = yields.index[-1] - pd.DateOffset(months=6)
    recent = yields[yields.index >= six_months_ago]
    recent_slope = float(recent.iloc[-1] - recent.iloc[0]) if len(recent) > 1 else 0.0

    window = f"{start_yield:.2f}% → {end_yield:.2f}%"
    if change_bps <= -50 or recent_slope <= -0.25:
        return "FAVORABLE", (f"India 10Y yields are down roughly {abs(change_bps):.0f} bps over "
                             f"the 5-year window ({window}). Falling yields lower the discount "
                             f"rate on future cash flows, which tends to support equity valuations.")
    if change_bps >= 100 or recent_slope >= 0.5:
        return "UNFAVORABLE", (f"India 10Y yields are up roughly {change_bps:.0f} bps over the "
                               f"5-year window ({window}). Rising yields raise the cost of capital "
                               f"and typically compress equity valuation multiples.")
    return "NEUTRAL", (f"India 10Y yields moved roughly {change_bps:+.0f} bps over the 5-year "
                       f"window ({window}) — a broadly stable-rate backdrop for equities.")


def fetch_macro_factors():
    """Daily close series for each macro factor that Yahoo returns."""
    key = "macro_factors"
    cached = cache_get(key)
    if cached is not None:
        return cached
    out = {}
    for name, ticker in MACRO_FACTOR_TICKERS.items():
        df = fetch_price_history_cached(ticker)
        if not df.empty and "Close" in df.columns and df["Close"].notna().sum() > 200:
            out[name] = df["Close"].dropna()
    if len(out) == len(MACRO_FACTOR_TICKERS):   # only cache a complete set
        cache_set(key, out)
    return out


def market_context():
    """Everything the pages need about the market & rates, in one call."""
    rf = fetch_risk_free()
    bench_df = fetch_benchmark()
    if bench_df is None:
        return None
    bench_close = adjusted_close(bench_df)
    bench_returns = bench_close.pct_change().dropna()
    years = (bench_close.index[-1] - bench_close.index[0]).days / 365.25
    market_cagr = float((bench_close.iloc[-1] / bench_close.iloc[0]) ** (1 / years) - 1)
    stance, note = macro_yield_commentary(rf["yields"])
    return {
        "rf": rf["rate"], "rf_source": rf["source"], "rf_as_of": rf["as_of"],
        "rf_yields": rf["yields"],
        "bench_close": bench_close, "bench_returns": bench_returns,
        "market_cagr": market_cagr,
        "market_vol": float(bench_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "macro_stance": stance, "macro_note": note,
    }


# ==============================================================================
# CHART HELPERS
# ==============================================================================

def new_figure(figsize=(9, 4.4), dpi=140, nrows=1, ncols=1):
    fig = Figure(figsize=figsize, dpi=dpi, facecolor=CHART_BG)
    axes = fig.subplots(nrows, ncols)
    for ax in np.atleast_1d(axes).ravel():
        style_axis(ax)
    return fig, axes


def style_axis(ax):
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=CHART_DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(CHART_SPINE)
    ax.grid(alpha=0.15, color=CHART_GRID)
    ax.xaxis.label.set_color(CHART_DIM)
    ax.yaxis.label.set_color(CHART_DIM)


def set_title(ax, text):
    ax.set_title(text, fontsize=12.5, color=CHART_TEXT, weight="bold", loc="left")


def legend(ax, loc="upper left", ncol=1):
    leg = ax.legend(loc=loc, fontsize=8, frameon=False, labelcolor="#e6e6e6", ncol=ncol)
    return leg


def fig_to_b64(fig):
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
