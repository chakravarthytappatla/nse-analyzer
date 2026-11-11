/* Page 01 · Descriptive Statistics */
(function () {
  const { escapeHtml: esc, num, img } = NS;
  const resultsGrid = document.getElementById('resultsGrid');
  const correlationSection = document.getElementById('correlationSection');
  const correlationTable = document.getElementById('correlationTable');
  let explanations = {};

  const note = (key) => NS.note(explanations[key]);
  const v = (x, suffix = '') => (x === null || x === undefined ? '—' : x + suffix);

  NS.init({
    minAnalyze: 1,
    maxAnalyze: 6,
    statusMessages: [
      'Pulling five years of price history…',
      'Reading the India 10-year G-Sec yield…',
      'Running beta regression against Nifty 50…',
      'Scoring Sharpe, Treynor, Sortino, Alpha and Fama…',
      'Rendering charts…',
    ],
    async run(symbols) {
      resultsGrid.innerHTML = '';
      correlationSection.hidden = true;
      const data = await NS.postJSON('/api/analyze', { symbols });
      explanations = data.explanations || {};
      NS.renderMacro(data);
      renderResults(data.results);
      if (data.correlation) {
        correlationTable.innerHTML = NS.corrTable(data.correlation);
        correlationSection.hidden = false;
      }
    },
  });

  function gradeItem(label, value, grade, key) {
    return `
      <div class="grade-item">
        <div class="g-label">${label}</div>
        <div class="g-value">${value}</div>
        <div class="g-tag ${esc(grade)}">${esc(grade)}</div>
        ${note(key)}
      </div>`;
  }

  function renderRangeBar(r) {
    const { S3: s3, R3: r3 } = r.pivots;
    const pct = Math.min(100, Math.max(0, ((r.price - s3) / ((r3 - s3) || 1)) * 100));
    return `
      <div class="range-bar-label"><span>S3 · ₹${r.pivots.S3}</span><span>R3 · ₹${r.pivots.R3}</span></div>
      <div class="range-bar"><div class="marker" style="left:${pct}%" data-price="₹${r.price}"></div></div>
      <div class="range-legend"><span>S1 ₹${r.pivots.S1}</span><span>Pivot ₹${r.pivots.P}</span><span>R1 ₹${r.pivots.R1}</span></div>`;
  }

  function renderResults(results) {
    resultsGrid.innerHTML = '';
    results.forEach((r) => {
      const card = document.createElement('div');
      card.className = 'card';
      if (r.error) {
        card.innerHTML = `<div class="card-head"><h3>${esc(r.ticker)}</h3></div><p class="card-error">${esc(r.error)}</p>`;
        resultsGrid.appendChild(card);
        return;
      }
      const outClass = r.outperformance_pct >= 0 ? 'pos' : 'neg';
      const outSign = r.outperformance_pct >= 0 ? '+' : '';
      const normal = NS.isNum(r.jb_p) ? (r.jb_p < 0.05 ? 'Non-normal' : 'Normal') : '—';

      card.innerHTML = `
        <div class="card-head">
          <h3>${esc(r.ticker)}</h3>
          <span class="badge ${esc(r.valuation.replace(' ', '.'))}" data-val="${esc(r.valuation)}">${esc(r.valuation)}</span>
        </div>
        <div class="price-row">
          <span class="price">₹${r.price.toLocaleString('en-IN')}</span>
          <span class="sub">10-DMA ₹${r.dma10.toLocaleString('en-IN')} · RSI ${v(r.rsi)} (${esc(r.rsi_zone)})</span>
        </div>
        ${note('valuation')}

        <div class="chart-block">${img(r.chart_b64, `${r.ticker} price chart with 10-DMA and pivot levels`)}</div>
        ${renderRangeBar(r)}
        ${note('pivots')}

        <div class="benchmark-strip">
          <div class="bstat"><span class="b-label">${esc(r.ticker)}, annualised</span><span class="b-value">${v(r.annual_return_pct, '%')}</span></div>
          <div class="bstat"><span class="b-label">Nifty 50, annualised</span><span class="b-value">${v(r.benchmark_annual_return_pct, '%')}</span></div>
          <div class="bstat"><span class="b-label">Outperformance</span><span class="b-value ${outClass}">${outSign}${v(r.outperformance_pct, '%')}</span></div>
        </div>
        ${note('outperformance')}
        <div class="chart-block">${img(r.benchmark_chart_b64, `${r.ticker} vs Nifty 50 growth comparison`)}</div>
        ${note('benchmark_chart')}

        <div class="grades">
          ${gradeItem('Sharpe', v(r.sharpe), r.sharpe_grade, 'sharpe')}
          ${gradeItem('Sortino', v(r.sortino), r.sortino_grade, 'sortino')}
          ${gradeItem('Treynor', v(r.treynor), r.treynor_grade, 'treynor')}
          ${gradeItem('Jensen Alpha', v(r.alpha_pct, '%'), r.alpha_grade, 'alpha')}
          ${gradeItem('Fama net', v(r.fama_pct, '%'), r.fama_grade, 'fama')}
        </div>

        <div class="chart-block">${img(r.histogram_b64, `${r.ticker} daily return distribution histogram`)}</div>
        ${note('histogram')}

        <div class="stat-grid">
          <div><span class="stat-label">Beta</span>${v(r.beta)}${note('beta')}</div>
          <div><span class="stat-label">R² vs Nifty</span>${v(r.r_squared)}${note('r_squared')}</div>
          <div><span class="stat-label">Ann. return</span>${v(r.annual_return_pct, '%')}${note('annual_return')}</div>
          <div><span class="stat-label">CAGR</span>${v(r.cagr_pct, '%')}${note('cagr')}</div>
          <div><span class="stat-label">Ann. vol</span>${v(r.annual_vol_pct, '%')}${note('annual_vol')}</div>
          <div><span class="stat-label">Max drawdown</span>${v(r.max_drawdown_pct, '%')}${note('max_drawdown')}</div>
          <div><span class="stat-label">Skew</span>${v(r.skewness)}${note('skewness')}</div>
          <div><span class="stat-label">Excess kurtosis</span>${v(r.kurtosis)}${note('kurtosis')}</div>
          <div><span class="stat-label">Jarque-Bera</span>${normal} (p ${NS.pval(r.jb_p)})${note('jarque_bera')}</div>
          <div><span class="stat-label">VaR 95% (1-day)</span>${v(r.var_95_pct, '%')}${note('var_95_pct')}</div>
          <div><span class="stat-label">Net asset risk</span>${v(r.net_asset_risk_pct)}${note('net_asset_risk_pct')}</div>
          <div><span class="stat-label">Mean / Median daily</span>${v(r.mean_daily_pct, '%')} / ${v(r.median_daily_pct, '%')}</div>
        </div>`;
      resultsGrid.appendChild(card);
    });
  }
})();
