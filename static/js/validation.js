/* Page 04 · Model Validation — walk-forward backtest 2018 → today */
(function () {
  const { escapeHtml: esc, num, rupee, pval, short, img, isNum } = NS;
  const list = document.getElementById('validationList');
  let runId = 0;
  const store = {};               // ticker -> result (for CSV download)
  const p1 = (v) => (isNum(v) ? v.toFixed(1) + '%' : '—');
  const p2 = (v) => (isNum(v) ? v.toFixed(2) + '%' : '—');

  NS.init({
    minAnalyze: 1,
    maxAnalyze: 6,
    statusMessages: [
      'Downloading daily prices since 2013…',
      'Re-fitting ARIMA + GARCH for every year, using only past data…',
      'Forecasting each trading day it has never seen…',
      'Scoring against real prices and the naive forecast…',
    ],
    async run(symbols) {
      const myRun = ++runId;
      list.innerHTML = '';
      const slots = {};
      symbols.forEach((t) => {
        const el = document.createElement('article');
        el.className = 'card wide';
        el.innerHTML = `<div class="card-head"><h3>${esc(t)}</h3></div><p class="muted">Queued…</p>`;
        list.appendChild(el);
        slots[t] = el;
      });
      const queue = [...symbols];
      const worker = async () => {
        while (queue.length) {
          const t = queue.shift();
          const r = await pollBacktest(t, slots[t], myRun);
          if (myRun !== runId) return;
          if (!r.error) store[r.ticker] = r;
          slots[t].innerHTML = card(r);
          const btn = slots[t].querySelector('.csv-btn');
          if (btn) btn.addEventListener('click', () => downloadCsv(r.ticker));
        }
      };
      // Fire every stock at once: the server queues them and computes one at a time.
      await Promise.all(symbols.map(() => worker()));
    },
  });

  // The server answers 202 {status: queued|running} until the job is done,
  // then 200 with the result. Polling keeps every request short, well under
  // the cloud host's request time limit.
  const POLL_MS = 4000;
  const MAX_WAIT_MS = 30 * 60 * 1000;
  const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

  async function pollBacktest(ticker, slot, myRun) {
    const t0 = Date.now();
    let failures = 0;
    while (Date.now() - t0 < MAX_WAIT_MS) {
      if (myRun !== runId) return { ticker, error: 'Cancelled — a newer run started.' };
      let r;
      try {
        r = await NS.postJSON('/api/backtest', { symbol: ticker });
        failures = 0;
      } catch (e) {
        if (++failures >= 4) return { ticker, error: e.message };
        await sleep(POLL_MS);      // transient 502/503 while the host restarts
        continue;
      }
      if (!r.status) return r;     // finished: the full result
      const msg = slot.querySelector('.muted');
      if (msg) {
        msg.textContent = r.status === 'queued'
          ? `Queued — ${r.queue_position} stock(s) ahead. Each new stock takes ~2–3 min the first time; saved results load instantly after.`
          : `Re-fitting ARIMA + GARCH for each year 2018 → today… ${r.elapsed}s elapsed (first run ~2–3 min; finished years are saved).`;
      }
      await sleep(POLL_MS);
    }
    return { ticker, error: 'Timed out after 30 minutes — please try again.' };
  }

  // ------------------------------------------------------------ verdicts
  function check(label, pass, detail, neutral = false) {
    const cls = neutral ? 'neutral' : pass ? 'pass' : 'fail';
    const icon = neutral ? '≈' : pass ? '✓' : '✗';
    return `<div class="check ${cls}"><span class="c-icon" aria-hidden="true">${icon}</span><div><b>${label}</b><span>${detail}</span></div></div>`;
  }

  function verdicts(s) {
    const u = s.theil_u;
    const priceEdge = u < 0.98;
    const priceSame = u >= 0.98 && u <= 1.02;
    return [
      check('Price level vs naive forecast', priceEdge,
        `Theil's U ${num(u, 3)} — ${priceEdge ? 'beats' : priceSame ? 'ties with' : 'loses to'} “tomorrow = today”. Accuracy ${p2(s.accuracy)} looks high, but the naive forecast scores almost the same.`,
        priceSame),
      check('Direction (up / down)', s.hit_rate > 52,
        `Hit-rate ${p1(s.hit_rate)} (coin-flip = 50%) · Spearman ρ ${num(s.spearman, 3)}. Daily direction is not predictable — consistent with an efficient market.`,
        s.hit_rate >= 48 && s.hit_rate <= 52),
      check('90% price range is honest', s.coverage90 >= 85 && s.coverage90 <= 95,
        `Real price fell inside the 1-day 90% band ${p1(s.coverage90)} of the time, and inside the 21-day band ${p1(s.coverage21)} (target 90%).`),
      check('95% VaR (risk) is accurate', isNum(s.kupiec_p) && s.kupiec_p > 0.05,
        `Breached on ${p1(s.var_breach)} of days (target 5%) · Kupiec p ${pval(s.kupiec_p)} ${s.kupiec_p > 0.05 ? '→ statistically OK' : '→ rejected'}.`),
      check('Volatility forecast vs naive', s.vol_ratio < 1,
        `GARCH error is ${num(s.vol_ratio, 2)}× the naive (last-21-day) volatility error${s.vol_ratio < 1 ? ' — it forecasts risk better' : ''}.`),
      check('No overfitting', s.overfit_ratio >= 0.8 && s.overfit_ratio <= 1.2,
        `Out-of-sample ÷ in-sample error = ${num(s.overfit_ratio, 2)} (≈ 1 means it works as well on unseen data).`),
    ].join('');
  }

  // --------------------------------------------------------------- card
  function card(r) {
    if (r.error) return `<div class="card-head"><h3>${esc(r.ticker)}</h3></div><p class="card-error">${esc(r.error)}</p>`;
    const s = r.summary;
    const ex = r.explanations;
    const years = r.years.filter((y) => !y.error);

    const priceRows = years.map((y) => `
      <tr><th>${y.year}${y.year === new Date().getFullYear() ? ' <span class="tag">YTD</span>' : ''}<div class="spec">${y.test_from} → ${y.test_to} · ${y.days} days</div></th>
        <td>${esc(y.arima)}<div class="spec">${esc(y.garch)}</div></td>
        <td class="hl">${p2(y.accuracy)}</td><td>${p2(y.mape)}</td><td>${p2(y.rmspe)}</td>
        <td>${rupee(y.rmse)}</td><td>${rupee(y.rw_rmse)}</td>
        <td class="${y.theil_u < 1 ? 'pos' : 'neg'}">${num(y.theil_u, 3)}</td>
        <td>${p1(y.hit_rate)}</td><td>${isNum(y.spearman) ? num(y.spearman, 3) : "n/a*"}</td><td>${num(y.r2_os, 3)}</td></tr>`).join('');

    const riskRows = years.map((y) => `
      <tr><th>${y.year}</th>
        <td class="hl">${p2(y.accuracy21)}</td><td>${num(y.theil_u21, 3)}</td><td>${p1(y.hit21)}</td>
        <td class="${y.coverage21 >= 80 ? 'pos' : 'neg'}">${p1(y.coverage21)}</td>
        <td class="${y.coverage90 >= 85 && y.coverage90 <= 95 ? 'pos' : 'neg'}">${p1(y.coverage90)}</td>
        <td>${p1(y.var_breach)} <span class="muted">(${y.var_breaches})</span></td>
        <td class="${y.kupiec_p > 0.05 ? 'pos' : 'neg'}">${pval(y.kupiec_p)}</td>
        <td class="${y.vol_ratio < 1 ? 'pos' : 'neg'}">${num(y.vol_ratio, 2)}</td>
        <td>${num(y.overfit_ratio, 2)}</td></tr>`).join('');
    const failed = r.years.filter((y) => y.error).map((y) => `${y.year}: ${esc(y.error)}`).join(' · ');

    const f21 = r.forecasts21.map((f) => {
      const errPct = (f.pred / f.actual - 1) * 100;
      const inside = f.actual >= f.lo90 && f.actual <= f.hi90;
      return `<tr><th>${f.date}</th><td>${f.origin}</td><td>${rupee(f.p0)}</td><td>${rupee(f.pred)}</td>
        <td>${rupee(f.lo90, 0)} – ${rupee(f.hi90, 0)}</td><td>${rupee(f.actual)}</td>
        <td>${(errPct >= 0 ? '+' : '') + errPct.toFixed(2)}%</td><td>${inside ? '✓ inside' : '✗ outside'}</td></tr>`;
    }).reverse().join('');

    return `
      <div class="card-head">
        <div><h3>${esc(r.ticker)}</h3><div class="card-sub">${s.days.toLocaleString('en-IN')} out-of-sample trading days · ${s.n21} monthly 21-day forecasts · ${r.first_year} → ${years.at(-1).test_to}</div></div>
        <div class="headline-num"><span class="h-label">1-day price accuracy</span><span class="h-value">${p2(s.accuracy)}</span></div>
      </div>

      <div class="kpi-row">
        <div class="kpi"><span class="k-label">1-day accuracy · MAPE</span><span class="k-value">${p2(s.accuracy)}</span><span class="k-sub">MAPE ${p2(s.mape)} · RMSPE ${p2(s.rmspe)}</span></div>
        <div class="kpi"><span class="k-label">21-day accuracy</span><span class="k-value">${p2(s.accuracy21)}</span><span class="k-sub">Theil's U ${num(s.theil_u21, 3)}</span></div>
        <div class="kpi"><span class="k-label">Theil's U (1-day)</span><span class="k-value">${num(s.theil_u, 3)}</span><span class="k-sub">&lt; 1 beats naive</span></div>
        <div class="kpi"><span class="k-label">Direction hit-rate</span><span class="k-value">${p1(s.hit_rate)}</span><span class="k-sub">21-day: ${p1(s.hit21)}</span></div>
      </div>

      <div class="check-grid">${verdicts(s)}</div>
      ${NS.note(ex.accuracy)}

      ${img(r.price_chart_b64, `${r.ticker} predicted vs actual price, ${r.first_year} to today`)}
      ${img(r.vol_chart_b64, `${r.ticker} predicted vs realised volatility`)}

      <h4>Year by year · 1-day price forecasts</h4>
      <div class="table-wrap"><table class="data-table compact">
        <thead><tr><th>Test year</th><th>Models (picked by AIC)</th><th>Accuracy</th><th>MAPE</th><th>RMSPE</th><th>RMSE</th><th>Naive RMSE</th><th>Theil's U</th><th>Hit-rate</th><th>Spearman ρ</th><th>R² (OOS)</th></tr></thead>
        <tbody>${priceRows}</tbody></table></div>
      ${NS.note(ex.theil)}${NS.note(ex.r2)}<p class="metric-note">* n/a = ARIMA(0,0,0) predicts the same average return every day, so a rank correlation is undefined.</p>

      <h4>Year by year · 21-day forecasts, risk &amp; overfitting</h4>
      <div class="table-wrap"><table class="data-table compact">
        <thead><tr><th>Test year</th><th>21d accuracy</th><th>21d Theil's U</th><th>21d hit-rate</th><th>21d 90% coverage</th><th>1d 90% coverage</th><th>VaR 95% breaches</th><th>Kupiec p</th><th>Vol error vs naive</th><th>Overfit ratio</th></tr></thead>
        <tbody>${riskRows}</tbody></table></div>
      ${failed ? `<p class="muted">Skipped: ${failed}</p>` : ''}
      ${NS.note(ex.coverage)}${NS.note(ex.var)}${NS.note(ex.vol)}${NS.note(ex.overfit)}

      <details class="coef-details"><summary>Real vs predicted — all ${r.forecasts21.length} monthly 21-day forecasts (newest first)</summary>
        <div class="table-wrap"><table class="data-table compact">
          <thead><tr><th>Target date</th><th>Forecast made</th><th>Price then</th><th>Predicted</th><th>90% range</th><th>Actual</th><th>Error</th><th>In range?</th></tr></thead>
          <tbody>${f21}</tbody></table></div>
      </details>
      <button type="button" class="btn-secondary csv-btn">Download all ${s.days.toLocaleString('en-IN')} daily real vs predicted prices (CSV)</button>`;
  }

  // ----------------------------------------------------------------- CSV
  function downloadCsv(ticker) {
    const d = store[ticker].daily;
    const lines = ['date,actual,predicted,naive_forecast,lower_90,upper_90,error_pct,inside_90'];
    d.date.forEach((dt, i) => {
      const err = ((d.pred[i] / d.actual[i] - 1) * 100).toFixed(3);
      const inside = d.actual[i] >= d.lo90[i] && d.actual[i] <= d.hi90[i] ? 1 : 0;
      lines.push([dt, d.actual[i], d.pred[i], d.naive[i], d.lo90[i], d.hi90[i], err, inside].join(','));
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${short(ticker)}_backtest_daily.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
})();
