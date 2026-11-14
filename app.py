"""
==============================================================================
NSE TERMINAL — Flask backend (routes only; maths lives in the page modules)
==============================================================================
Pages
  GET  /                    -> Page 1 · Descriptive Statistics
  GET  /wacc                -> Page 2 · WACC Analysis
  GET  /portfolio           -> Page 3 · Portfolio Management
  GET  /validation          -> Page 4 · Model Validation (2018 → today backtest)
API
  GET  /api/suggest?q=TC    -> NSE ticker autocomplete
  POST /api/analyze         -> descriptive statistics      (descriptive.py)
  POST /api/wacc            -> cost of capital + DCF        (wacc.py)
  POST /api/portfolio       -> optimisation + backtest      (portfolio.py)
  POST /api/forecast        -> ARIMA / GARCH forecast, 1 stock (forecast.py)
  POST /api/backtest        -> walk-forward validation, 1 stock (backtest.py)
==============================================================================
"""
import os
import threading
import time

import requests
from flask import Flask, jsonify, render_template, request

import backtest
import descriptive
import forecast
import portfolio
import wacc
from core import as_float, cache_get, cache_set, clean_json, market_context, normalize_symbols

app = Flask(__name__)


def _error(msg, code=400):
    return jsonify({"error": msg}), code


def _context_or_error():
    ctx = market_context()
    if ctx is None:
        return None, _error("Could not reach Yahoo Finance for Nifty 50 benchmark data right now. "
                            "Please try again shortly.", 502)
    return ctx, None


def _macro_payload(ctx):
    return {
        "risk_free_rate_pct": round(ctx["rf"] * 100, 2),
        "bond_source": ctx["rf_source"], "rf_as_of": ctx["rf_as_of"],
        "macro_stance": ctx["macro_stance"], "macro_note": ctx["macro_note"],
        "market_cagr_pct": round(ctx["market_cagr"] * 100, 2),
    }


def _pct_arg(payload, key, default, lo, hi):
    """Read a percentage from the request (e.g. 7.25) → decimal, clamped."""
    v = as_float(payload.get(key))
    if v is None:
        return default
    return min(max(v / 100, lo), hi)


# ==============================================================================
# PAGES
# ==============================================================================

@app.route("/")
def page_descriptive():
    return render_template("descriptive.html", active="descriptive")


@app.route("/wacc")
def page_wacc():
    return render_template("wacc.html", active="wacc",
                           defaults={"erp": wacc.DEFAULT_ERP * 100, "tax": wacc.DEFAULT_TAX_RATE * 100,
                                     "g": wacc.DEFAULT_TERMINAL_GROWTH * 100})


@app.route("/portfolio")
def page_portfolio():
    return render_template("portfolio.html", active="portfolio",
                           defaults={"max_weight": portfolio.DEFAULT_MAX_WEIGHT * 100,
                                     "erp": portfolio.DEFAULT_ERP * 100})


# ==============================================================================
# API
# ==============================================================================

@app.route("/validation")
def page_validation():
    return render_template("validation.html", active="validation",
                           first_year=backtest.FIRST_TEST_YEAR)


@app.route("/api/suggest")
def api_suggest():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    key = f"suggest:{q.lower()}"
    cached = cache_get(key)
    if cached is not None:
        return jsonify(cached)
    try:
        resp = requests.get("https://query1.finance.yahoo.com/v1/finance/search",
                            params={"q": q, "quotesCount": 10, "newsCount": 0},
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        quotes = resp.json().get("quotes", [])
    except Exception:
        quotes = []
    results, seen = [], set()
    for item in quotes:
        symbol = item.get("symbol", "")
        if not symbol.endswith(".NS") or symbol in seen:
            continue
        seen.add(symbol)
        results.append({"symbol": symbol, "name": item.get("longname") or item.get("shortname") or symbol})
    cache_set(key, results)
    return jsonify(results)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    payload = request.get_json(silent=True) or {}
    symbols = normalize_symbols(payload.get("symbols"), descriptive.MAX_TICKERS)
    if not symbols:
        return _error("No valid symbols supplied.")
    ctx, err = _context_or_error()
    if err:
        return err
    out = descriptive.run_descriptive(symbols, ctx)
    out.update(_macro_payload(ctx))
    return jsonify(clean_json(out))


@app.route("/api/wacc", methods=["POST"])
def api_wacc():
    payload = request.get_json(silent=True) or {}
    symbols = normalize_symbols(payload.get("symbols"), wacc.MAX_TICKERS)
    if not symbols:
        return _error("No valid symbols supplied.")
    ctx, err = _context_or_error()
    if err:
        return err
    erp = _pct_arg(payload, "erp", wacc.DEFAULT_ERP, 0.0, 0.20)
    tax = _pct_arg(payload, "tax_rate", wacc.DEFAULT_TAX_RATE, 0.0, 0.50)
    g = _pct_arg(payload, "terminal_growth", wacc.DEFAULT_TERMINAL_GROWTH, -0.02, 0.10)
    out = wacc.run_wacc(symbols, ctx, erp, tax, g)
    out.update(_macro_payload(ctx))
    return jsonify(clean_json(out))


@app.route("/api/portfolio", methods=["POST"])
def api_portfolio():
    payload = request.get_json(silent=True) or {}
    symbols = normalize_symbols(payload.get("symbols"), portfolio.MAX_TICKERS)
    if len(symbols) < portfolio.MIN_TICKERS:
        return _error("Add at least 2 companies to build a portfolio.")
    ctx, err = _context_or_error()
    if err:
        return err
    max_w = _pct_arg(payload, "max_weight", portfolio.DEFAULT_MAX_WEIGHT, 0.05, 1.0)
    erp = _pct_arg(payload, "erp", portfolio.DEFAULT_ERP, 0.0, 0.20)
    amount = as_float(payload.get("amount"), 1_000_000)
    amount = min(max(amount, 10_000), 1e11)
    try:
        out = portfolio.run_portfolio(symbols, ctx, max_w, amount, erp)
    except Exception as exc:
        return _error(f"Portfolio analysis failed: {exc}", 500)
    if "error" in out:
        return _error(out["error"], 422)
    out.update(_macro_payload(ctx))
    return jsonify(clean_json(out))


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    payload = request.get_json(silent=True) or {}
    symbols = normalize_symbols([payload.get("symbol")], 1)
    if not symbols:
        return _error("No valid symbol supplied.")
    try:
        out = forecast.run_forecast(symbols[0])
    except Exception as exc:
        out = {"ticker": symbols[0], "error": f"Forecast failed: {exc}"}
    return jsonify(clean_json(out))


# ------------------------------------------------------------------------------
# Backtest = background job + polling. A first run of a stock can take minutes
# on a small cloud CPU, longer than Azure's ~230 s request limit, so the
# request never waits for it: it starts (or reports) a job and the page polls.
# Jobs run one at a time so they don't fight over a single-core plan.
# ------------------------------------------------------------------------------
_jobs = {}                                # symbol -> {"status", "result", "started"}
_jobs_lock = threading.Lock()
_compute_slot = threading.Semaphore(1)    # one backtest computes at a time
JOB_RESULT_TTL = 15 * 60


def _run_backtest_job(symbol):
    with _compute_slot:
        with _jobs_lock:
            _jobs[symbol]["status"] = "running"
            _jobs[symbol]["started"] = time.time()
        try:
            result = clean_json(backtest.run_backtest(symbol))
        except Exception as exc:
            result = {"ticker": symbol, "error": f"Backtest failed: {exc}"}
    with _jobs_lock:
        _jobs[symbol].update(status="done", result=result, finished=time.time())


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    payload = request.get_json(silent=True) or {}
    symbols = normalize_symbols([payload.get("symbol")], 1)
    if not symbols:
        return _error("No valid symbol supplied.")
    symbol = symbols[0]
    with _jobs_lock:
        job = _jobs.get(symbol)
        if job and job["status"] == "done":
            result = job["result"]
            if "error" in result:
                # report a failure once, then forget it so the next click retries
                del _jobs[symbol]
                return jsonify(result)
            if time.time() - job["finished"] < JOB_RESULT_TTL:
                return jsonify(result)
            job = None
        if job is None:
            job = {"status": "queued", "queued_at": time.time()}
            _jobs[symbol] = job
            threading.Thread(target=_run_backtest_job, args=(symbol,), daemon=True).start()
        ahead = sum(1 for s, j in _jobs.items()
                    if j["status"] in ("queued", "running") and s != symbol
                    and j.get("queued_at", 0) < job.get("queued_at", 0))
        elapsed = time.time() - job.get("started", time.time())
    return jsonify({"ticker": symbol, "status": job["status"], "queue_position": ahead,
                    "elapsed": round(elapsed)}), 202


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8790))
    app.run(host="0.0.0.0", port=port)
