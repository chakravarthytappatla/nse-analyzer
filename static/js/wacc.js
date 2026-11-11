/* Page 02 · WACC Analysis */
(function () {
  const { escapeHtml: esc, num, pct, signedPct, rupee, crore, pval, short, img } = NS;
  const summarySection = document.getElementById('summarySection');
  const summaryTable = document.getElementById('summaryTable');
  const resultsList = document.getElementById('resultsList');
  const readPct = (id) => parseFloat(document.getElementById(id).value);
  let ex = {};
  const note = (k) => NS.note(ex[k]);

  NS.init({
    minAnalyze: 1,
    maxAnalyze: 6,
    statusMessages: [
      'Resampling to weekly returns…',
      'Racing CAPM vs Dimson vs macro MLR (AIC / BIC)…',
      'Pulling balance sheet, income statement & cash flows…',
      'Rating credit from interest coverage…',
      'Discounting free cash flows at WACC…',
    ],
    async run(symbols) {
      summarySection.hidden = true;
      resultsList.innerHTML = '';
      const data = await NS.postJSON('/api/wacc', {
        symbols, erp: readPct('erpInput'), tax_rate: readPct('taxInput'), terminal_growth: readPct('growthInput'),
      });
      ex = data.explanations || {};
      NS.renderMacro(data, [['Equity risk premium (input)', pct(data.assumptions.erp), 'editable above']]);
      renderSummary(data.results);
      data.results.forEach((r) => resultsList.appendChild(renderCard(r)));
    },
  });

  function renderSummary(results) {
    const ok = results.filter((r) => !r.error);
    if (!ok.length) return;
    const rows = ok.map((r) => `
      <tr>
        <th>${esc(short(r.ticker))}</th>
        <td>${num(r.adj_beta)}</td><td>${pct(r.ke)}</td><td>${pct(r.kd_post)}</td>
        <td>${pct(r.wd, 1)}</td><td class="hl">${r.is_financial ? 'n/m' : pct(r.wacc)}</td>
        <td>${r.dcf.ok ? rupee(r.dcf.per_share, 0) : '—'}</td>
        <td class="${r.dcf.ok ? (r.dcf.upside >= 0 ? 'pos' : 'neg') : ''}">${r.dcf.ok ? signedPct(r.dcf.upside) : '—'}</td>
      </tr>`).join('');
    summaryTable.innerHTML = `
      <table class="data-table">
        <thead><tr><th>Stock</th><th>Adj. β</th><th>Ke</th><th>Kd after tax</th><th>D / V</th><th>WACC</th><th>DCF value</th><th>vs price</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="metric-note">n/m = not meaningful (financial company — use Ke as the hurdle rate).</p>`;
    summarySection.hidden = false;
  }

  function ledgerRow(label, formula, value, cls = '') {
    return `<div class="ledger-row ${cls}"><span class="l-label">${label}</span><span class="l-formula">${formula}</span><span class="l-value">${value}</span></div>`;
  }

  function modelTable(r) {
    const head = '<tr><th>Model</th><th>R²</th><th>Adj. R²</th><th>AIC</th><th>BIC</th><th>F-test p</th><th>DW</th><th>JB p</th><th>BP p</th><th>Max VIF</th><th>Market β</th></tr>';
    const rows = r.models.map((m) => {
      const tags = [m.best_aic ? '<span class="tag tag-accent">Best AIC</span>' : '',
        m.best_bic ? '<span class="tag">Best BIC</span>' : '',
        m.beta_source ? '<span class="tag tag-info">β used</span>' : ''].join('');
      return `<tr class="${m.best_aic ? 'row-best' : ''}">
        <th><div>${esc(m.name)}</div><div class="spec">${esc(m.spec)}</div><div>${tags}</div></th>
        <td>${num(m.r2, 3)}</td><td>${num(m.adj_r2, 3)}</td><td>${num(m.aic, 1)}</td><td>${num(m.bic, 1)}</td>
        <td>${pval(m.f_pvalue)}</td><td>${num(m.dw)}</td><td>${pval(m.jb_p)}</td><td>${pval(m.bp_p)}</td>
        <td>${num(m.max_vif)}</td><td>${num(m.market_beta, 3)}</td></tr>`;
    }).join('');
    const coefs = r.models.map((m) => `
      <details class="coef-details"><summary>${esc(m.name)} — coefficients (Newey-West HAC t-stats, n = ${m.n})</summary>
        <table class="data-table compact"><thead><tr><th>Term</th><th>Coef.</th><th>t</th><th>p</th><th></th></tr></thead><tbody>
        ${m.coefs.map((c) => `<tr><th>${esc(c.name)}</th><td>${num(c.coef, 4)}</td><td>${num(c.t)}</td><td>${pval(c.p)}</td>
          <td>${c.p < 0.05 ? '<span class="tag tag-accent">significant</span>' : '<span class="tag">n.s.</span>'}</td></tr>`).join('')}
        </tbody></table></details>`).join('');
    return `<div class="table-wrap"><table class="data-table">${head}${rows}</table></div>${coefs}`;
  }

  function dcfBlock(r) {
    const d = r.dcf;
    if (!d.ok) return `<div class="subsection"><h4>DCF valuation</h4><p class="muted">${esc(d.reason)}</p></div>`;
    const flows = d.flows.map((f) => `<tr><th>Year ${f.year}</th><td>${pct(f.growth, 1)}</td><td>${crore(f.fcf)}</td><td>${crore(f.pv)}</td></tr>`).join('');
    const s = d.sensitivity;
    const grid = s.values.map((row, i) => `<tr><th>${pct(s.wacc[i], 1)}</th>${row.map((v, j) => {
      if (v === null) return '<td>—</td>';
      const up = v / r.price - 1;
      const centre = i === 2 && j === 2 ? ' centre' : '';
      return `<td class="${up >= 0 ? 'pos' : 'neg'}${centre}">${rupee(v, 0)}</td>`;
    }).join('')}</tr>`).join('');
    return `
      <div class="subsection">
        <h4>DCF valuation (two-stage FCFF)</h4>
        <div class="kpi-row">
          <div class="kpi"><span class="k-label">Intrinsic value / share</span><span class="k-value">${rupee(d.per_share, 0)}</span></div>
          <div class="kpi"><span class="k-label">Market price</span><span class="k-value">${rupee(r.price, 0)}</span></div>
          <div class="kpi"><span class="k-label">Upside / downside</span><span class="k-value ${d.upside >= 0 ? 'pos' : 'neg'}">${signedPct(d.upside)}</span></div>
          <div class="kpi"><span class="k-label">Terminal value share of EV</span><span class="k-value">${pct(d.tv_share, 0)}</span></div>
        </div>
        ${note('dcf')}
        <p class="metric-note">Base FCF = 3-yr average ${crore(d.base_fcf)} · stage-1 growth ${pct(d.stage1_growth, 1)} (revenue CAGR ${pct(d.hist_revenue_cagr, 1)}, capped 0–15%) fading to ${pct(d.terminal_growth, 1)} · EV ${crore(d.ev)} − debt + cash = equity ${crore(d.equity)}.</p>
        <div class="two-col">
          <div class="table-wrap"><table class="data-table compact"><thead><tr><th></th><th>Growth</th><th>FCF</th><th>PV</th></tr></thead><tbody>${flows}
            <tr><th>Terminal</th><td>${pct(d.terminal_growth, 1)}</td><td>—</td><td>${crore(d.pv_terminal)}</td></tr></tbody></table></div>
          <div class="table-wrap"><table class="data-table compact sens"><thead><tr><th>WACC ↓ / g →</th>${s.growth.map((g) => `<th>${pct(g, 1)}</th>`).join('')}</tr></thead><tbody>${grid}</tbody></table>
            <p class="metric-note">Value per share across WACC × terminal growth. Green = above today's market price, red = below. Outlined cell = base case.</p></div>
        </div>
      </div>`;
  }

  function renderCard(r) {
    const card = document.createElement('article');
    card.className = 'card wide';
    if (r.error) {
      card.innerHTML = `<div class="card-head"><h3>${esc(r.ticker)}</h3></div><p class="card-error">${esc(r.error)}</p>`;
      return card;
    }
    const warn = r.warnings.length ? `<ul class="warn-list">${r.warnings.map((w) => `<li>${esc(w)}</li>`).join('')}</ul>` : '';
    const coverage = NS.isNum(r.coverage) ? `${num(r.coverage, 1)}×` : 'no interest expense';
    card.innerHTML = `
      <div class="card-head">
        <div><h3>${esc(r.ticker)}</h3><div class="card-sub">${esc(r.name)} · ${esc(r.sector || '—')} · ${esc(r.industry || '')}</div></div>
        <div class="headline-num"><span class="h-label">${r.is_financial ? 'Cost of equity' : 'WACC'}</span><span class="h-value">${pct(r.is_financial ? r.ke : r.wacc)}</span></div>
      </div>
      ${warn}

      <div class="two-col">
        <div class="subsection">
          <h4>WACC build-up</h4>
          <div class="ledger">
            ${ledgerRow('Risk-free rate', 'India 10Y G-Sec', pct(r.rf))}
            ${ledgerRow('Beta (raw → Blume)', `${num(r.raw_beta, 3)} × 0.67 + 0.33`, num(r.adj_beta, 3))}
            ${ledgerRow('Equity risk premium', `input · historical ${pct(r.hist_erp)}`, pct(r.erp))}
            ${ledgerRow('Cost of equity Ke', 'Rf + β × ERP', pct(r.ke), 'sum')}
            ${ledgerRow('Synthetic rating', `coverage ${coverage}`, `${esc(r.rating)} (+${pct(r.spread)})`)}
            ${ledgerRow('Cost of debt Kd', `Rf + spread · observed ${pct(r.observed_kd)}`, pct(r.kd_pre))}
            ${ledgerRow('After-tax Kd', `Kd × (1 − ${pct(r.tax_rate)}) · effective tax ${pct(r.effective_tax, 1)}`, pct(r.kd_post), 'sum')}
            ${ledgerRow('Weights', `E ${crore(r.market_cap)} · D ${crore(r.debt)}`, `${pct(r.we, 1)} / ${pct(r.wd, 1)}`)}
            ${ledgerRow('WACC', 'E/V·Ke + D/V·Kd(1−t)', pct(r.wacc), 'total')}
          </div>
          <div class="cap-bar" role="img" aria-label="Capital structure: equity ${pct(r.we, 1)}, debt ${pct(r.wd, 1)}">
            <span class="cap-e" style="width:${(r.we * 100).toFixed(2)}%"></span><span class="cap-d" style="width:${(r.wd * 100).toFixed(2)}%"></span>
          </div>
          <div class="cap-legend"><span><i class="sw sw-e"></i>Equity ${pct(r.we, 1)}</span><span><i class="sw sw-d"></i>Debt ${pct(r.wd, 1)}</span><span>Net debt ${crore(r.net_debt)}</span></div>
          ${note('ke')}${note('kd')}${note('wacc')}
        </div>
        <div class="subsection">
          <h4>Security characteristic line</h4>
          ${img(r.regression_chart_b64, `${r.ticker} weekly regression vs Nifty 50`)}
          ${img(r.rolling_beta_chart_b64, `${r.ticker} rolling 52-week beta`)}
          ${note('rolling_beta')}
        </div>
      </div>

      <div class="subsection">
        <h4>Beta model race · ${r.weeks} weekly observations${r.factors_used.length ? ' · macro factors: ' + esc(r.factors_used.join(', ')) : ''}</h4>
        ${note('beta_race')}
        ${modelTable(r)}
        <p class="verdict">Best fit by AIC: <b>${esc(r.best_aic)}</b> · by BIC: <b>${esc(r.best_bic)}</b> · WACC beta from <b>${esc(r.beta_model)}</b> (best AIC among market-only models, since CAPM needs a total market beta).${r.stepwise_note ? ' ' + esc(r.stepwise_note) : ''}</p>
        ${note('diagnostics')}${note('blume')}
      </div>

      ${r.is_financial ? '' : dcfBlock(r)}`;
    return card;
  }
})();
