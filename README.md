# TERMINAL.NS

A small web app that turns the NSE quant/macro/technical analysis script into
a searchable dashboard:

- Type a company name → autocomplete suggests NSE tickers (via Yahoo Finance's
  own search endpoint, filtered to `.NS` symbols) — the `.NS` suffix is always
  applied automatically, so people just type "TCS" or "Reliance".
- Add up to 6 companies to a watchlist and run one combined analysis.
- Get macro commentary (dynamic 10Y bond yield read), per-stock stats,
  Sharpe / Treynor / Jensen's Alpha / Fama grades, RSI, pivot levels, a price
  chart, and a correlation matrix if you compared more than one stock.

## Project layout

```
nse-analyzer/
├── app.py                 # Flask backend + all analysis logic
├── requirements.txt
├── Procfile                # for gunicorn-based hosts (Render, Railway, Heroku)
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

## Deploy

### Option A — Render.com (free tier friendly)
1. Push this folder to a GitHub repo.
2. On Render: **New → Web Service** → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --workers 2 --threads 4 --timeout 120`
5. Deploy. Render auto-detects the `Procfile` too, so the start command can be
   left blank if you prefer.

### Option B — Railway.app
1. `railway init` in this folder, then `railway up`.
2. Railway reads the `Procfile` automatically.

### Option C — Any VPS (systemd + gunicorn + nginx)
```bash
pip install -r requirements.txt
gunicorn app:app --bind 0.0.0.0:8000 --workers 2 --threads 4 --timeout 120
```
Put nginx in front as a reverse proxy to `127.0.0.1:8000` and terminate TLS there.

### Option D — Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "120"]
```

## Notes & limits

- Yahoo Finance can rate-limit or occasionally rename/retire tickers (this bit
  the earlier `^IN10YT=X` bond-yield symbol) — the backend already tries a
  short list of fallback symbols and a static rate as a last resort, and
  caches successful lookups for 15 minutes to reduce load.
- `matplotlib` runs headless (`Agg` backend) and returns charts as base64 PNG,
  so no extra static file handling is needed for images.
- `MAX_TICKERS_PER_REQUEST` in `app.py` (default 6) caps how many symbols can
  be analyzed in one request — raise it if your host has more CPU/time budget,
  since each ticker triggers its own price-history download and chart render.
- This is a research tool, not investment advice — the UI footer says so too.
