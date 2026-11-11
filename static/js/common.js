/* ==========================================================================
   common.js — shared by all three pages
   Search/autocomplete, a watchlist that persists across pages (localStorage),
   loader + error handling, macro banner and number formatting helpers.
   ========================================================================== */
(function () {
  const STORAGE_KEY = 'ns_terminal_watchlist';
  const WATCHLIST_CAP = 10;

  const $ = (id) => document.getElementById(id);
  const input = $('tickerInput');
  const suggestList = $('suggestList');
  const watchlistEl = $('watchlist');
  const analyzeBtn = $('analyzeBtn');
  const watchlistHint = $('watchlistHint');
  const statusArea = $('statusArea');
  const statusText = $('statusText');
  const errorArea = $('errorArea');
  const macroBanner = $('macroBanner');

  let config = { minAnalyze: 1, maxAnalyze: 6, statusMessages: ['Working…'], run: async () => {} };
  let watchlist = loadWatchlist();
  let activeSuggestions = [];
  let activeIndex = -1;
  let debounceTimer = null;

  // ------------------------------------------------------------------ utils
  function escapeHtml(str) {
    return String(str ?? '')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const num = (v, d = 2) => (isNum(v) ? v.toFixed(d) : '—');
  const pct = (v, d = 2) => (isNum(v) ? (v * 100).toFixed(d) + '%' : '—');
  const signedPct = (v, d = 1) => (isNum(v) ? (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + '%' : '—');
  const rupee = (v, d = 2) => (isNum(v) ? '₹' + v.toLocaleString('en-IN', { maximumFractionDigits: d, minimumFractionDigits: d }) : '—');
  function crore(v) {
    if (!isNum(v)) return '—';
    return '₹' + (v / 1e7).toLocaleString('en-IN', { maximumFractionDigits: 0 }) + ' Cr';
  }
  const pval = (p) => (!isNum(p) ? '—' : p < 0.001 ? '<0.001' : p.toFixed(3));
  const short = (t) => String(t).replace(/\.NS$/, '');
  const img = (b64, alt) => (b64 ? `<div class="chart-wrap"><img src="data:image/png;base64,${b64}" alt="${escapeHtml(alt)}" loading="lazy" /></div>` : '');
  const note = (text) => (text ? `<p class="metric-note">${escapeHtml(text)}</p>` : '');

  async function postJSON(url, body) {
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    let data = {};
    try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status}).`);
    return data;
  }

  // -------------------------------------------------------- watchlist store
  function loadWatchlist() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
      return Array.isArray(raw) ? raw.filter((w) => w && w.symbol).slice(0, WATCHLIST_CAP) : [];
    } catch (_) { return []; }
  }
  function saveWatchlist() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(watchlist)); } catch (_) { /* private mode */ }
  }

  // ------------------------------------------------------------ autocomplete
  input.addEventListener('input', () => {
    const q = input.value.trim();
    clearTimeout(debounceTimer);
    if (!q) { hideSuggestions(); return; }
    debounceTimer = setTimeout(() => fetchSuggestions(q), 220);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && suggestList.hidden && input.value.trim()) {
      e.preventDefault();
      addToWatchlist({ symbol: input.value.trim().toUpperCase(), name: input.value.trim() });
      return;
    }
    if (suggestList.hidden) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive(Math.min(activeIndex + 1, activeSuggestions.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(Math.max(activeIndex - 1, 0)); }
    else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIndex >= 0 && activeSuggestions[activeIndex]) addToWatchlist(activeSuggestions[activeIndex]);
      else if (input.value.trim()) addToWatchlist({ symbol: input.value.trim().toUpperCase(), name: input.value.trim() });
    } else if (e.key === 'Escape') hideSuggestions();
  });

  document.addEventListener('click', (e) => {
    if (!suggestList.contains(e.target) && e.target !== input) hideSuggestions();
  });

  async function fetchSuggestions(q) {
    try {
      const res = await fetch(`/api/suggest?q=${encodeURIComponent(q)}`);
      renderSuggestions(await res.json());
    } catch (_) { renderSuggestions([]); }
  }

  function renderSuggestions(items) {
    activeSuggestions = items;
    activeIndex = -1;
    suggestList.innerHTML = '';
    if (!items.length) {
      suggestList.innerHTML = '<li class="suggest-empty">No NSE match found — press Enter to add it anyway.</li>';
    } else {
      items.forEach((item, i) => {
        const li = document.createElement('li');
        li.setAttribute('role', 'option');
        li.innerHTML = `<span class="suggest-name">${escapeHtml(item.name)}</span><span class="suggest-symbol">${escapeHtml(item.symbol)}</span>`;
        li.addEventListener('click', () => addToWatchlist(item));
        li.addEventListener('mouseenter', () => setActive(i));
        suggestList.appendChild(li);
      });
    }
    suggestList.hidden = false;
    input.setAttribute('aria-expanded', 'true');
  }

  function setActive(i) {
    activeIndex = i;
    [...suggestList.children].forEach((li, idx) => li.classList.toggle('active', idx === i));
  }
  function hideSuggestions() {
    suggestList.hidden = true;
    input.setAttribute('aria-expanded', 'false');
  }

  // --------------------------------------------------------------- watchlist
  function addToWatchlist(item) {
    if (watchlist.length >= WATCHLIST_CAP) {
      watchlistHint.textContent = `The watchlist holds up to ${WATCHLIST_CAP} companies.`;
      return;
    }
    if (!watchlist.some((w) => w.symbol === item.symbol)) watchlist.push(item);
    saveWatchlist();
    renderWatchlist();
    input.value = '';
    hideSuggestions();
    input.focus();
  }

  function removeFromWatchlist(symbol) {
    watchlist = watchlist.filter((w) => w.symbol !== symbol);
    saveWatchlist();
    renderWatchlist();
  }

  function renderWatchlist() {
    watchlistEl.innerHTML = '';
    watchlist.forEach((item, i) => {
      const chip = document.createElement('div');
      chip.className = 'chip' + (i >= config.maxAnalyze ? ' chip-muted' : '');
      chip.innerHTML = `<span>${escapeHtml(item.symbol)}</span>`;
      const btn = document.createElement('button');
      btn.setAttribute('aria-label', `Remove ${item.symbol}`);
      btn.textContent = '×';
      btn.addEventListener('click', () => removeFromWatchlist(item.symbol));
      chip.appendChild(btn);
      watchlistEl.appendChild(chip);
    });
    const n = watchlist.length;
    analyzeBtn.disabled = n < config.minAnalyze;
    if (n === 0) watchlistHint.textContent = 'Add a company to get started';
    else if (n < config.minAnalyze) watchlistHint.textContent = `Add at least ${config.minAnalyze} companies`;
    else if (n > config.maxAnalyze) watchlistHint.textContent = `This page analyses the first ${config.maxAnalyze} of ${n}`;
    else watchlistHint.textContent = `${n} of ${config.maxAnalyze} added`;
  }

  // ------------------------------------------------------------------- run
  analyzeBtn.addEventListener('click', async () => {
    errorArea.hidden = true;
    macroBanner.hidden = true;
    statusArea.hidden = false;
    analyzeBtn.disabled = true;
    let i = 0;
    statusText.textContent = config.statusMessages[0];
    const timer = setInterval(() => {
      i = (i + 1) % config.statusMessages.length;
      statusText.textContent = config.statusMessages[i];
    }, 1500);
    try {
      await config.run(watchlist.slice(0, config.maxAnalyze).map((w) => w.symbol));
    } catch (err) {
      showError(err.message || 'Could not reach the analysis service. Please try again.');
    } finally {
      clearInterval(timer);
      statusArea.hidden = true;
      analyzeBtn.disabled = watchlist.length < config.minAnalyze;
    }
  });

  function showError(msg) {
    errorArea.textContent = msg;
    errorArea.hidden = false;
  }

  function renderMacro(data, extra = []) {
    const stats = [
      ['Risk-free rate (India 10Y)', `${num(data.risk_free_rate_pct)}%`, data.rf_as_of ? `as of ${data.rf_as_of}` : ''],
      ['Nifty 50 CAGR, 5y', `${num(data.market_cagr_pct)}%`, ''],
      ...extra,
    ];
    macroBanner.innerHTML = `
      ${stats.map(([l, v, s]) => `
        <div class="macro-stat"><span class="label">${escapeHtml(l)}</span><span class="value">${v}</span>
        ${s ? `<span class="sub">${escapeHtml(s)}</span>` : ''}</div>`).join('')}
      <div class="macro-note">
        <span class="macro-tag ${escapeHtml(data.macro_stance)}">${escapeHtml(data.macro_stance)}</span>
        <div>${escapeHtml(data.macro_note)}<div class="source">Source: ${escapeHtml(data.bond_source || '—')}</div></div>
      </div>`;
    macroBanner.hidden = false;
  }

  function corrTable(correlation) {
    const { tickers, matrix } = correlation;
    const color = (v) => (v >= 0 ? `rgba(57,135,229,${Math.min(0.55, v * 0.55)})` : `rgba(217,89,38,${Math.min(0.55, -v * 0.55)})`);
    let html = '<table class="corr"><thead><tr><th></th>';
    tickers.forEach((t) => (html += `<th>${escapeHtml(short(t))}</th>`));
    html += '</tr></thead><tbody>';
    matrix.forEach((row, i) => {
      html += `<tr><th>${escapeHtml(short(tickers[i]))}</th>`;
      row.forEach((v) => (html += `<td style="background:${color(v)}">${num(v)}</td>`));
      html += '</tr>';
    });
    return html + '</tbody></table>';
  }

  window.NS = {
    init(cfg) { config = { ...config, ...cfg }; renderWatchlist(); },
    postJSON, showError, renderMacro, corrTable,
    escapeHtml, isNum, num, pct, signedPct, rupee, crore, pval, short, img, note,
  };
})();
