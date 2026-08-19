(function () {
  const input = document.getElementById('tickerInput');
  const suggestList = document.getElementById('suggestList');
  const watchlistEl = document.getElementById('watchlist');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const watchlistHint = document.getElementById('watchlistHint');
  const statusArea = document.getElementById('statusArea');
  const statusText = document.getElementById('statusText');
  const errorArea = document.getElementById('errorArea');
  const macroBanner = document.getElementById('macroBanner');
  const resultsGrid = document.getElementById('resultsGrid');
  const correlationSection = document.getElementById('correlationSection');
  const correlationTable = document.getElementById('correlationTable');

  const MAX_WATCHLIST = 6;
  let watchlist = [];      // [{symbol, name}]
  let activeSuggestions = [];
  let activeIndex = -1;
  let debounceTimer = null;
  let explanations = {};   // filled in from /api/analyze response each run

  const STATUS_MESSAGES = [
    'Pulling five years of price history…',
    'Reading the current 10-year bond yield…',
    'Running beta regression against Nifty 50…',
    'Scoring Sharpe, Treynor, Sortino, Alpha and Fama…',
    'Rendering charts…',
  ];

  // ------------------------------------------------------------------
  // Autocomplete
  // ------------------------------------------------------------------

  input.addEventListener('input', () => {
    const q = input.value.trim();
    clearTimeout(debounceTimer);
    if (q.length === 0) {
      hideSuggestions();
      return;
    }
    debounceTimer = setTimeout(() => fetchSuggestions(q), 220);
  });

  input.addEventListener('keydown', (e) => {
    if (suggestList.hidden) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive(Math.min(activeIndex + 1, activeSuggestions.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive(Math.max(activeIndex - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIndex >= 0 && activeSuggestions[activeIndex]) {
        addToWatchlist(activeSuggestions[activeIndex]);
      } else if (input.value.trim()) {
        addToWatchlist({ symbol: input.value.trim().toUpperCase(), name: input.value.trim() });
      }
    } else if (e.key === 'Escape') {
      hideSuggestions();
    }
  });

  document.addEventListener('click', (e) => {
    if (!suggestList.contains(e.target) && e.target !== input) hideSuggestions();
  });

  async function fetchSuggestions(q) {
    try {
      const res = await fetch(`/api/suggest?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      renderSuggestions(data);
    } catch (err) {
      renderSuggestions([]);
    }
  }

  function renderSuggestions(items) {
    activeSuggestions = items;
    activeIndex = -1;
    suggestList.innerHTML = '';

    if (items.length === 0) {
      const li = document.createElement('li');
      li.className = 'suggest-empty';
      li.textContent = 'No NSE match found — press Enter to add it anyway.';
      suggestList.appendChild(li);
      suggestList.hidden = false;
      input.setAttribute('aria-expanded', 'true');
      return;
    }

    items.forEach((item, i) => {
      const li = document.createElement('li');
      li.setAttribute('role', 'option');
      li.innerHTML = `<span class="suggest-name">${escapeHtml(item.name)}</span><span class="suggest-symbol">${escapeHtml(item.symbol)}</span>`;
      li.addEventListener('click', () => addToWatchlist(item));
      li.addEventListener('mouseenter', () => setActive(i));
      suggestList.appendChild(li);
    });
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

  // ------------------------------------------------------------------
  // Watchlist
  // ------------------------------------------------------------------

  function addToWatchlist(item) {
    if (watchlist.length >= MAX_WATCHLIST) {
      watchlistHint.textContent = `You can compare up to ${MAX_WATCHLIST} companies at a time.`;
      return;
    }
    if (watchlist.some((w) => w.symbol === item.symbol)) {
      input.value = '';
      hideSuggestions();
      return;
    }
    watchlist.push(item);
    renderWatchlist();
    input.value = '';
    hideSuggestions();
    input.focus();
  }

  function removeFromWatchlist(symbol) {
    watchlist = watchlist.filter((w) => w.symbol !== symbol);
    renderWatchlist();
  }

  function renderWatchlist() {
    watchlistEl.innerHTML = '';
    watchlist.forEach((item) => {
      const chip = document.createElement('div');
      chip.className = 'chip';
      chip.innerHTML = `<span>${escapeHtml(item.symbol)}</span>`;
      const btn = document.createElement('button');
      btn.setAttribute('aria-label', `Remove ${item.symbol}`);
      btn.textContent = '×';
      btn.addEventListener('click', () => removeFromWatchlist(item.symbol));
      chip.appendChild(btn);
      watchlistEl.appendChild(chip);
    });

    analyzeBtn.disabled = watchlist.length === 0;
    watchlistHint.textContent = watchlist.length === 0
      ? 'Add a company to get started'
      : `${watchlist.length} of ${MAX_WATCHLIST} added`;
  }

  // ------------------------------------------------------------------
  // Analyze
  // ------------------------------------------------------------------

  analyzeBtn.addEventListener('click', runAnalysis);

  async function runAnalysis() {
    errorArea.hidden = true;
    macroBanner.hidden = true;
    resultsGrid.innerHTML = '';
    correlationSection.hidden = true;
    statusArea.hidden = false;

    let msgIdx = 0;
    statusText.textContent = STATUS_MESSAGES[0];
    const statusTimer = setInterval(() => {
      msgIdx = (msgIdx + 1) % STATUS_MESSAGES.length;
      statusText.textContent = STATUS_MESSAGES[msgIdx];
    }, 1400);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: watchlist.map((w) => w.symbol) }),
      });
      const data = await res.json();

      clearInterval(statusTimer);
      statusArea.hidden = true;

      if (!res.ok) {
        showError(data.error || 'Something went wrong while running the analysis.');
        return;
      }

      explanations = data.explanations || {};
      renderMacro(data);
      renderResults(data.results);
      renderCorrelation(data.correlation);
    } catch (err) {
      clearInterval(statusTimer);
      statusArea.hidden = true;
      showError('Could not reach the analysis service. Please try again.');
    }
  }

  function showError(msg) {
    errorArea.textContent = msg;
    errorArea.hidden = false;
  }

  function note(key) {
    const text = explanations[key];
    return text ? `<div class="metric-note">${escapeHtml(text)}</div>` : '';
  }

  function renderMacro(data) {
    macroBanner.innerHTML = `
      <div class="macro-stat">
        <span class="label">Risk-free rate</span>
        <span class="value">${data.risk_free_rate_pct}%</span>
      </div>
      <div class="macro-stat">
        <span class="label">Nifty 50, annualised</span>
        <span class="value">${data.market_annual_return_pct}%</span>
      </div>
      <div class="macro-note">
        <span class="macro-tag ${data.macro_stance}">${data.macro_stance}</span>
        <div>${escapeHtml(data.macro_note)}</div>
      </div>
    `;
    macroBanner.hidden = false;
  }

  function renderResults(results) {
    resultsGrid.innerHTML = '';
    results.forEach((r) => {
      const card = document.createElement('div');
      card.className = 'card';

      if (r.error) {
        card.innerHTML = `
          <div class="card-head"><h3>${escapeHtml(r.ticker)}</h3></div>
          <p class="card-error">${escapeHtml(r.error)}</p>
        `;
        resultsGrid.appendChild(card);
        return;
      }

      const valClass = r.valuation.replace(' ', '.');
      const outClass = r.outperformance_pct >= 0 ? 'pos' : 'neg';
      const outSign = r.outperformance_pct >= 0 ? '+' : '';

      card.innerHTML = `
        <div class="card-head">
          <h3>${escapeHtml(r.ticker)}</h3>
          <span class="badge ${valClass}" data-val="${r.valuation}">${r.valuation}</span>
        </div>
        <div class="price-row">
          <span class="price">₹${r.price.toLocaleString('en-IN')}</span>
          <span class="sub">10-DMA ₹${r.dma10.toLocaleString('en-IN')} · RSI ${r.rsi ?? '—'} (${r.rsi_zone})</span>
        </div>
        ${note('valuation')}

        <div class="chart-block">
          <div class="chart-wrap"><img src="data:image/png;base64,${r.chart_b64}" alt="${escapeHtml(r.ticker)} price chart with 10-DMA and pivot levels" /></div>
        </div>
        ${renderRangeBar(r)}
        ${note('pivots')}

        <div class="benchmark-strip">
          <div class="bstat">
            <span class="b-label">${escapeHtml(r.ticker)}, annualised</span>
            <span class="b-value">${r.annual_return_pct}%</span>
          </div>
          <div class="bstat">
            <span class="b-label">Nifty 50, annualised</span>
            <span class="b-value">${r.benchmark_annual_return_pct}%</span>
          </div>
          <div class="bstat">
            <span class="b-label">Outperformance</span>
            <span class="b-value ${outClass}">${outSign}${r.outperformance_pct}%</span>
          </div>
        </div>
        ${note('outperformance')}
        <div class="chart-block">
          ${r.benchmark_chart_b64
            ? `<div class="chart-wrap"><img src="data:image/png;base64,${r.benchmark_chart_b64}" alt="${escapeHtml(r.ticker)} vs Nifty 50 growth comparison" /></div>`
            : ''}
        </div>
        ${note('benchmark_chart')}

        <div class="grades">
          ${gradeItem('Sharpe', r.sharpe, r.sharpe_grade, 'sharpe')}
          ${gradeItem('Sortino', r.sortino ?? '—', r.sortino_grade, 'sortino')}
          ${gradeItem('Treynor', r.treynor, r.treynor_grade, 'treynor')}
          ${gradeItem('Jensen Alpha', r.alpha_pct + '%', r.alpha_grade, 'alpha')}
          ${gradeItem('Fama net', r.fama_pct + '%', r.fama_grade, 'fama')}
        </div>

        <div class="chart-block">
          <div class="chart-wrap"><img src="data:image/png;base64,${r.histogram_b64}" alt="${escapeHtml(r.ticker)} daily return distribution histogram" /></div>
        </div>
        ${note('histogram')}

        <div class="stat-grid">
          <div><span class="stat-label">Beta</span>${r.beta}${note('beta')}</div>
          <div><span class="stat-label">Ann. return</span>${r.annual_return_pct}%${note('annual_return')}</div>
          <div><span class="stat-label">Ann. vol</span>${r.annual_vol_pct}%${note('annual_vol')}</div>
          <div><span class="stat-label">Max drawdown</span>${r.max_drawdown_pct}%${note('max_drawdown')}</div>
          <div><span class="stat-label">Skew</span>${r.skewness}${note('skewness')}</div>
          <div><span class="stat-label">Kurtosis</span>${r.kurtosis}${note('kurtosis')}</div>
        </div>
      `;
      resultsGrid.appendChild(card);
    });
  }

  function gradeItem(label, value, grade, key) {
    return `
      <div class="grade-item">
        <div class="g-label">${label}</div>
        <div class="g-value">${value}</div>
        <div class="g-tag ${grade}">${grade}</div>
        ${note(key)}
      </div>
    `;
  }

  function renderRangeBar(r) {
    const s3 = r.pivots.S3, r3 = r.pivots.R3, price = r.price;
    const span = r3 - s3 || 1;
    const pct = Math.min(100, Math.max(0, ((price - s3) / span) * 100));
    return `
      <div class="range-bar-label"><span>S3 · ₹${r.pivots.S3}</span><span>R3 · ₹${r.pivots.R3}</span></div>
      <div class="range-bar"><div class="marker" style="left:${pct}%" data-price="₹${price}"></div></div>
      <div class="range-legend"><span>S1 ₹${r.pivots.S1}</span><span>Pivot ₹${r.pivots.P}</span><span>R1 ₹${r.pivots.R1}</span></div>
    `;
  }

  function renderCorrelation(correlation) {
    if (!correlation) {
      correlationSection.hidden = true;
      return;
    }
    const { tickers, matrix } = correlation;
    let html = '<table class="corr"><thead><tr><th></th>';
    tickers.forEach((t) => (html += `<th>${escapeHtml(t)}</th>`));
    html += '</tr></thead><tbody>';
    matrix.forEach((row, i) => {
      html += `<tr><th>${escapeHtml(tickers[i])}</th>`;
      row.forEach((v) => {
        const color = corrColor(v);
        html += `<td style="background:${color}">${v.toFixed(2)}</td>`;
      });
      html += '</tr>';
    });
    html += '</tbody></table>';
    correlationTable.innerHTML = html;
    correlationSection.hidden = false;
  }

  function corrColor(v) {
    // -1 -> red, 0 -> transparent, 1 -> green, subtle
    if (v >= 0) return `rgba(55,214,122,${Math.min(0.5, v * 0.5)})`;
    return `rgba(255,92,92,${Math.min(0.5, -v * 0.5)})`;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
})();
