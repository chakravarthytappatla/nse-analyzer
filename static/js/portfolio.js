/* Page 03 · Portfolio Management Analysis */
(function () {
  const { escapeHtml: esc, num, pct, signedPct, rupee, pval, short, img, isNum } = NS;
  const out = document.getElementById('portfolioResults');
  const forecastSection = document.getElementById('forecastSection');
  const forecastList = document.getElementById('forecastList');
  const val = (id) => parseFloat(document.getElementById(id).value);
  let ex = {};
  let runId = 0;
  const note = (k) => NS.note(ex[k]);

  NS.init({
    minAnalyze: 2,
    maxAnalyze: 10,
    statusMessages: [
      'Aligning five years of adjusted prices…',
      'Shrinking the covariance matrix (Ledoit-Wolf)…',
      'Walk-forward backtest: re-optimising every quarter…',
      'Tracing the efficient frontier…',
      'Simulating 5,000 one-year paths…',
    ],
    async run(symbols) {
      const myRun = ++runId;
      out.innerHTML = '';
      forecastList.innerHTML = '';
      forecastSection.hidden = true;
      const data = await NS.postJSON('/api/portfolio', {
        symbols, amount: val('amountInput'), max_weight: val('maxWeightInput'), erp: val('erpInput'),
      });
      ex = data.explanations || {};
      NS.renderMacro(data, [['Out-of-sample window', `${data.settings.oos_start} → ${data.settings.oos_end}`, `${data.settings.cost_bps} bps cost per turnover`]]);
      render(data);
      runForecasts(data.tickers, myRun);
    },
  });

  // ----------------------------------------------------------- portfolio
  function render(d) {
    const best = d.leaderboard.find((l) => l.selected);
    const b = d.benchmark;
    const s = d.settings;
    const capNote = s.max_weight > s.requested_max_weight
      ? `<p class="metric-note">Max weight raised to ${pct(s.max_weight, 0)} — with ${d.tickers.length} stocks a lower cap cannot sum to 100%.</p>` : '';

    const lbRows = [...d.leaderboard].sort((a, b2) => (b2.sharpe ?? -9) - (a.sharpe ?? -9)).map((l, i) => `
      <tr class="${l.selected ? 'row-best' : ''}">
        <th>${i + 1}. ${esc(l.name)} ${l.selected ? '<span class="tag tag-accent">Selected</span>' : ''}</th>
        <td>${pct(l.cagr, 1)}</td><td>${pct(l.vol, 1)}</td><td>${num(l.sharpe)}</td><td>${num(l.sortino)}</td>
        <td>${pct(l.max_dd, 1)}</td><td>${num(l.calmar)}</td><td>${num(l.beta)}</td><td>${signedPct(l.jensen_alpha)}</td>
        <td>${num(l.info_ratio)}</td><td>${pct(l.avg_turnover, 0)}</td></tr>`).join('');

    const alloc = d.allocation.map((a) => `
      <tr><th>${esc(short(a.ticker))}</th>
        <td><div class="wbar"><span style="width:${(a.weight * 100).toFixed(1)}%"></span></div>${pct(a.weight, 1)}</td>
        <td><div class="wbar risk"><span style="width:${(a.risk_contribution * 100).toFixed(1)}%"></span></div>${pct(a.risk_contribution, 1)}</td>
        <td>${rupee(a.rupees, 0)}</td><td>${a.shares.toLocaleString('en-IN')} @ ${rupee(a.price)}</td></tr>`).join('');

    const strategies = Object.keys(d.weights.by_strategy);
    const wTable = `<table class="data-table compact"><thead><tr><th>Model</th>${d.tickers.map((t) => `<th>${esc(short(t))}</th>`).join('')}</tr></thead><tbody>
      ${strategies.map((st) => `<tr class="${st === d.selected ? 'row-best' : ''}"><th>${esc(st)}</th>${d.weights.by_strategy[st].map((w) => `<td>${w < 0.0005 ? '·' : pct(w, 1)}</td>`).join('')}</tr>`).join('')}
      </tbody></table>`;

    const assets = d.assets.map((a) => `<tr><th>${esc(short(a.ticker))}</th><td>${num(a.beta)}</td><td>${pct(a.ret, 1)}</td><td>${pct(a.ret_hist, 1)}</td><td>${pct(a.vol, 1)}</td></tr>`).join('');

    const amt = s.amount;
    const varRows = d.risk.var.map((v) => `
      <tr><th>${(v.confidence * 100).toFixed(0)}%</th>
        <td>${pct(v.historical_var)} <span class="muted">${rupee(v.historical_var * amt, 0)}</span></td>
        <td>${pct(v.normal_var)}</td><td>${pct(v.cornish_fisher_var)}</td>
        <td>${pct(v.historical_cvar)} <span class="muted">${rupee(v.historical_cvar * amt, 0)}</span></td>
        <td>${pct(v.normal_cvar)}</td></tr>`).join('');

    const mc = d.monte_carlo;
    out.innerHTML = `
      <article class="card wide">
        <div class="card-head">
          <div><h3>Selected model: ${esc(d.selected)}</h3>
            <div class="card-sub">Chosen because it had the highest out-of-sample Sharpe of the six models (${num(best.sharpe)} vs Nifty 50 ${num(b.sharpe)}).</div></div>
          <div class="headline-num"><span class="h-label">Out-of-sample CAGR</span><span class="h-value">${pct(best.cagr, 1)}</span></div>
        </div>
        ${capNote}
        ${(best.sharpe ?? -9) < (b.sharpe ?? -9) ? `<ul class="warn-list"><li>No model beat the Nifty 50 out-of-sample on a risk-adjusted basis for this basket. A low-cost Nifty index fund would have done better. Consider changing the stocks, not just the weights.</li></ul>` : ''}
        <div class="subsection">
          <h4>Model leaderboard · walk-forward, ${s.lookback_days}-day estimation window, rebalanced every ${s.rebalance_days} days</h4>
          ${note('backtest')}
          <div class="table-wrap"><table class="data-table">
            <thead><tr><th>Model</th><th>CAGR</th><th>Vol</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Calmar</th><th>β</th><th>Jensen α</th><th>Info ratio</th><th>Turnover / qtr</th></tr></thead>
            <tbody>${lbRows}
              <tr class="row-bench"><th>Nifty 50 (benchmark)</th><td>${pct(b.cagr, 1)}</td><td>${pct(b.vol, 1)}</td><td>${num(b.sharpe)}</td><td>${num(b.sortino)}</td><td>${pct(b.max_dd, 1)}</td><td>${num(b.calmar)}</td><td>1.00</td><td>—</td><td>—</td><td>—</td></tr>
            </tbody></table></div>
          ${img(d.backtest_chart_b64, 'Walk-forward backtest equity curves')}
        </div>
      </article>

      <article class="card wide">
        <div class="card-head"><div><h3>Allocation today</h3><div class="card-sub">${esc(d.selected)} fitted on the latest ${s.lookback_days} trading days · cap ${pct(s.max_weight, 0)} per stock · Ledoit-Wolf shrinkage ${pct(s.shrinkage, 1)}</div></div>
          <div class="headline-num"><span class="h-label">Amount</span><span class="h-value">${rupee(amt, 0)}</span></div></div>
        <div class="table-wrap"><table class="data-table">
          <thead><tr><th>Stock</th><th>Weight</th><th>Risk contribution</th><th>Rupees</th><th>Shares (whole)</th></tr></thead><tbody>${alloc}</tbody></table></div>
        ${note('risk_contribution')}
        <div class="kpi-row">
          <div class="kpi"><span class="k-label">Expected return (CAPM)</span><span class="k-value">${pct(d.risk.expected_return_capm, 1)}</span></div>
          <div class="kpi"><span class="k-label">Volatility</span><span class="k-value">${pct(d.risk.vol, 1)}</span></div>
          <div class="kpi"><span class="k-label">Portfolio beta</span><span class="k-value">${num(d.risk.beta)}</span></div>
          <div class="kpi"><span class="k-label">Diversification ratio</span><span class="k-value">${num(d.risk.diversification_ratio)}×</span></div>
        </div>
        <details class="coef-details"><summary>Weights under every model</summary><div class="table-wrap">${wTable}</div></details>
      </article>

      <article class="card wide">
        <div class="card-head"><div><h3>Efficient frontier</h3><div class="card-sub">Expected returns from CAPM (Rf + β × ERP) — historical means are shown for reference but are too noisy to optimise on directly.</div></div></div>
        ${img(d.frontier_chart_b64, 'Efficient frontier with capital market line')}${note('frontier')}
        <div class="table-wrap"><table class="data-table compact"><thead><tr><th>Stock</th><th>β</th><th>CAPM μ</th><th>Hist. μ (2y)</th><th>Vol</th></tr></thead><tbody>${assets}</tbody></table>${note('ledoit_wolf')}</div>
      </article>

      <article class="card wide">
        <div class="card-head"><div><h3>Risk</h3><div class="card-sub">1-day VaR / CVaR of today's allocation · skew ${num(d.risk.skew)} · excess kurtosis ${num(d.risk.excess_kurtosis)}</div></div></div>
        <div class="table-wrap"><table class="data-table">
          <thead><tr><th>Confidence</th><th>Historical VaR</th><th>Normal VaR</th><th>Cornish-Fisher VaR</th><th>Historical CVaR</th><th>Normal CVaR</th></tr></thead><tbody>${varRows}</tbody></table></div>
        ${note('var')}${note('cornish_fisher')}
        <div class="two-col">
          <div>${img(d.mc_chart_b64, 'Monte Carlo fan chart')}${note('monte_carlo')}</div>
          <div class="kpi-grid">
            <div class="kpi"><span class="k-label">Median in 1 year</span><span class="k-value">${rupee(mc.median, 0)}</span></div>
            <div class="kpi"><span class="k-label">5th – 95th percentile</span><span class="k-value small">${rupee(mc.p5, 0)} – ${rupee(mc.p95, 0)}</span></div>
            <div class="kpi"><span class="k-label">Probability of a loss</span><span class="k-value">${pct(mc.prob_loss, 1)}</span></div>
            <div class="kpi"><span class="k-label">1-yr VaR 95% / CVaR 95%</span><span class="k-value small">${rupee(mc.var_1y_95, 0)} / ${rupee(mc.cvar_1y_95, 0)}</span></div>
          </div>
        </div>
        <h4>Correlation (2-year window)</h4>
        <div class="table-wrap">${NS.corrTable(d.correlation)}</div>
      </article>`;
  }

  // ------------------------------------------------------------ forecasts
  async function runForecasts(tickers, myRun) {
    forecastSection.hidden = false;
    const slots = {};
    tickers.forEach((t) => {
      const el = document.createElement('article');
      el.className = 'card wide';
      el.innerHTML = `<div class="card-head"><h3>${esc(t)}</h3></div><p class="muted">Fitting ARIMA grid and GARCH models…</p>`;
      forecastList.appendChild(el);
      slots[t] = el;
    });
    const queue = [...tickers];
    const worker = async () => {
      while (queue.length) {
        const t = queue.shift();
        let r;
        try { r = await NS.postJSON('/api/forecast', { symbol: t }); } catch (e) { r = { ticker: t, error: e.message }; }
        if (myRun !== runId) return; // a newer run started
        slots[t].innerHTML = forecastCard(r);
      }
    };
    await Promise.all([worker(), worker()]);
  }

  function forecastCard(r) {
    if (r.error) return `<div class="card-head"><h3>${esc(r.ticker)}</h3></div><p class="card-error">${esc(r.error)}</p>`;
    const st = r.stationarity;
    const o = r.oos;
    const last = r.points[r.points.length - 1];
    const arima = r.arima.slice(0, 6).map((a) => `<tr class="${a.best_aic ? 'row-best' : ''}"><th>${esc(a.order)} ${a.best_aic ? '<span class="tag tag-accent">Best AIC</span>' : ''}${a.best_bic ? '<span class="tag">Best BIC</span>' : ''}</th><td>${num(a.aic, 1)}</td><td>${num(a.bic, 1)}</td><td>${num(a.llf, 1)}</td></tr>`).join('');
    const garch = r.garch.map((g) => `<tr class="${g.best_aic ? 'row-best' : ''}"><th>${esc(g.name)} ${g.best_aic ? '<span class="tag tag-accent">Best AIC</span>' : ''}</th><td>${num(g.aic, 1)}</td><td>${num(g.bic, 1)}</td><td>${num(g.alpha, 3)}</td><td>${num(g.beta, 3)}</td><td>${isNum(g.gamma) ? num(g.gamma, 3) : '—'}</td><td>${isNum(g.nu) ? num(g.nu, 1) : '—'}</td><td>${num(g.persistence, 3)}</td><td>${isNum(g.half_life_days) ? num(g.half_life_days, 0) + 'd' : '—'}</td></tr>`).join('');
    const pts = r.points.map((p) => `<tr><th>${p.days} days</th><td>${rupee(p.median, 0)}</td><td>${rupee(p.low90, 0)} – ${rupee(p.high90, 0)}</td><td>${pct(p.prob_up, 0)}</td></tr>`).join('');
    const verdict = o.beats_random_walk
      ? `beat the random walk by ${num(o.skill_vs_rw_pct, 2)}% RMSE — a small edge; treat with caution.`
      : `showed <b>no meaningful edge</b> over a random walk (${num(o.skill_vs_rw_pct, 2)}% RMSE skill) — consistent with weak-form market efficiency: daily direction is close to unpredictable, but volatility is.`;
    return `
      <div class="card-head">
        <div><h3>${esc(r.ticker)}</h3><div class="card-sub">${r.obs} daily log returns · mean model ${esc(r.best_arima)} · vol model ${esc(r.best_garch)}</div></div>
        <div class="headline-num"><span class="h-label">21-day median</span><span class="h-value">${rupee(last.median, 0)}</span></div>
      </div>
      <div class="kpi-row">
        <div class="kpi"><span class="k-label">Last price</span><span class="k-value">${rupee(r.last_price, 0)}</span></div>
        <div class="kpi"><span class="k-label">90% range, 21 days</span><span class="k-value small">${rupee(last.low90, 0)} – ${rupee(last.high90, 0)}</span></div>
        <div class="kpi"><span class="k-label">GARCH vol now → next 21d</span><span class="k-value small">${pct(r.current_vol, 1)} → ${pct(r.forecast_vol_21d, 1)}</span></div>
        <div class="kpi"><span class="k-label">P(price higher in 21d)</span><span class="k-value">${pct(last.prob_up, 0)}</span></div>
      </div>
      <p class="metric-note">ADF p: log price ${pval(st.adf_logprice_p)} (${st.adf_logprice_p > 0.05 ? 'unit root → prices are non-stationary' : 'stationary'}) · returns ${pval(st.adf_returns_p)} (${st.adf_returns_p < 0.05 ? 'stationary → model returns with d = 0' : 'non-stationary'}) · Jarque-Bera p ${pval(st.jb_p)}, excess kurtosis ${num(st.excess_kurtosis)} → fat tails, so Student-t GARCH is tested.</p>
      <div class="two-col">
        <div>${img(r.fan_chart_b64, `${r.ticker} 21-day forecast fan`)}</div>
        <div>${img(r.vol_chart_b64, `${r.ticker} GARCH conditional volatility`)}</div>
      </div>
      <div>
        <div class="table-wrap"><h4>Mean model · ARIMA grid (top 6 of 9 by AIC)</h4>
          <table class="data-table compact"><thead><tr><th>Order</th><th>AIC</th><th>BIC</th><th>Log-lik</th></tr></thead><tbody>${arima}</tbody></table>
          <p class="verdict">Out-of-sample (last ${o.test_days} days): RMSE ${num(o.rmse_model * 100, 3)}% vs random walk ${num(o.rmse_random_walk * 100, 3)}% vs hist. mean ${num(o.rmse_hist_mean * 100, 3)}% · direction hit-rate ${pct(o.hit_rate, 0)}. ${esc(r.best_arima)} ${verdict}</p>
        </div>
        <div class="table-wrap"><h4>Volatility model · GARCH family</h4>
          <table class="data-table compact"><thead><tr><th>Model</th><th>AIC</th><th>BIC</th><th>α</th><th>β</th><th>γ</th><th>ν</th><th>α+β+γ/2</th><th>Half-life</th></tr></thead><tbody>${garch}</tbody></table>
          <table class="data-table compact"><thead><tr><th>Horizon</th><th>Median</th><th>90% range</th><th>P(up)</th></tr></thead><tbody>${pts}</tbody></table>
        </div>
      </div>`;
  }
})();
