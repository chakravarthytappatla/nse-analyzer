/* ==========================================================================
   Page 04 · Smart Prediction — interactive price-probability explorer
   ========================================================================== */
(function () {
  const { escapeHtml: esc, isNum, short } = NS;
  const $ = (id) => document.getElementById(id);
  const area = $('predictArea');
  const tabs = $('stockTabs');
  const view = $('stockView');

  const state = { data: {}, order: [], current: null, horizon: '1d', targetPct: {}, auto: false };
  let refreshTimer = null;
  let runId = 0;
  const REFRESH_MS = 60 * 1000;
  const TZ = 'Asia/Kolkata';

  // ------------------------------------------------------------- formatting
  const inr = (v, d = 2) => (isNum(v) ? '₹' + v.toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d }) : '—');
  const pct = (p, d = 1) => (isNum(p) ? (p * 100).toFixed(d) + '%' : '—');
  const flat = (v, d = 2) => Math.abs(v) < 0.5 * Math.pow(10, -d);   // rounds to zero
  const signed = (v, d = 2) => (!isNum(v) ? '—' : flat(v, d) ? (0).toFixed(d) + '%' : (v > 0 ? '+' : '') + v.toFixed(d) + '%');
  const tone = (v) => (flat(v) ? '' : v > 0 ? 'pos' : 'neg');
  const fmt = (iso, opts) => new Intl.DateTimeFormat('en-IN', { timeZone: TZ, ...opts }).format(new Date(iso));
  const fmtWhen = (iso) => fmt(iso, { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false });
  const fmtDay = (iso) => fmt(iso, { day: 'numeric', month: 'short' });
  const fmtClock = (iso) => fmt(iso, { hour: '2-digit', minute: '2-digit', hour12: false });

  // ---------------------------------------------------------------- maths
  // erf: Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)
  function erf(x) {
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const normCdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  function logGamma(x) {           // Lanczos approximation
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x; const tmp = x + 5.5 - (x + 0.5) * Math.log(x + 5.5);
    let ser = 1.000000000190015;
    for (const ci of c) ser += ci / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }
  function betacf(a, b, x) {        // continued fraction for the incomplete beta
    const MAXIT = 200, EPS = 3e-14, FPMIN = 1e-300;
    let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d; let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; const del = d * c; h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }
  function ibeta(x, a, b) {         // regularised incomplete beta I_x(a, b)
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  function tCdf(t, nu) {
    const p = 0.5 * ibeta(nu / (nu + t * t), nu / 2, 0.5);
    return t > 0 ? 1 - p : p;
  }
  // CDF of a log-return with mean mu, st.dev sd, Student-t (unit variance) if nu
  function cdf(x, mu, sd, nu) {
    if (!(sd > 0)) return x >= mu ? 1 : 0;
    const z = (x - mu) / sd;
    return nu ? tCdf(z / Math.sqrt((nu - 2) / nu), nu) : normCdf(z);
  }
  const pAbove = (h, target) => 1 - cdf(Math.log(target / h.base), h.mu, h.sd, h.nu);
  // Probability the price touches the target at ANY time before the horizon
  // (Brownian motion with drift, reflection principle).
  function pTouch(h, target) {
    const b = Math.log(target / h.base), m = h.mu, s = h.sd;
    if (!(s > 0)) return 0;
    const k = Math.exp(Math.max(-50, Math.min(50, 2 * m * b / (s * s))));
    const p = b >= 0 ? normCdf((m - b) / s) + k * normCdf((-b - m) / s)
                     : normCdf((b - m) / s) + k * normCdf((b + m) / s);
    return Math.max(0, Math.min(1, p));
  }

  // ------------------------------------------------------------------ run
  NS.init({
    minAnalyze: 1,
    maxAnalyze: 6,
    statusMessages: [
      'Reading the latest 1-minute prices…',
      'Fitting ARIMA + GARCH on five years of history…',
      'Measuring minute-level and overnight volatility…',
      'Turning it into probabilities…',
    ],
    async run(symbols) {
      const myRun = ++runId;
      stopRefresh();
      state.data = {};
      state.order = symbols.map((s) => (s.endsWith('.NS') ? s : s + '.NS'));
      state.current = state.order[0];
      area.hidden = false;
      renderTabs();
      view.innerHTML = '<p class="muted">Predicting…</p>';
      const queue = [...state.order];
      const worker = async () => {
        while (queue.length) {
          const t = queue.shift();
          let r;
          try { r = await NS.postJSON('/api/predict', { symbol: t }); } catch (e) { r = { ticker: t, error: e.message }; }
          if (myRun !== runId) return;
          state.data[t] = r;
          renderTabs();
          if (t === state.current) renderStock();
        }
      };
      await Promise.all([worker(), worker()]);
    },
  });

  function renderTabs() {
    tabs.innerHTML = state.order.map((t) => {
      const d = state.data[t];
      const sub = !d ? 'loading…' : d.error ? 'error' : inr(d.price);
      return `<button type="button" role="tab" class="stock-tab ${t === state.current ? 'active' : ''}" aria-selected="${t === state.current}" data-t="${esc(t)}">
        <span class="st-name">${esc(short(t))}</span><span class="st-sub">${sub}</span></button>`;
    }).join('');
    tabs.querySelectorAll('.stock-tab').forEach((b) => b.addEventListener('click', () => {
      state.current = b.dataset.t;
      renderTabs();
      renderStock();
    }));
    tabs.hidden = state.order.length < 2;
  }

  // ------------------------------------------------------------ stock view
  const horizonOf = (d) => d.horizons.find((h) => h.key === state.horizon && !h.error) || d.horizons.find((h) => !h.error);
  const targetPrice = (d) => d.price * (1 + (state.targetPct[d.ticker] ?? 2) / 100);

  function renderStock() {
    const d = state.data[state.current];
    if (!d) { view.innerHTML = '<p class="muted">Predicting…</p>'; return; }
    if (d.error) { view.innerHTML = `<article class="card wide"><div class="card-head"><h3>${esc(d.ticker)}</h3></div><p class="card-error">${esc(d.error)}</p></article>`; return; }

    const status = d.market_open
      ? `<span class="live-badge"><span class="dot" aria-hidden="true"></span>Market open · last tick ${fmtClock(d.last_tick)} IST</span>`
      : `<span class="closed-badge">Market closed · opens ${fmtWhen(d.next_open)} IST</span>`;

    view.innerHTML = `
      <article class="card wide predict-card">
        <div class="card-head">
          <div>
            <h3>${esc(d.ticker)}</h3>
            <div class="card-sub">${status}</div>
          </div>
          <div class="headline-num"><span class="h-label">${d.market_open ? 'Live price' : `Last close · ${esc(d.last_close_date)}`}</span><span class="h-value">${inr(d.price)}</span></div>
        </div>

        <div class="horizon-bar" role="tablist" aria-label="Prediction horizon">
          ${d.horizons.map((h) => h.error ? '' : `
            <button type="button" role="tab" class="hz-btn" data-k="${h.key}">
              <span class="hz-label">${esc(h.label)}</span>
              <span class="hz-price">${inr(h.median)}</span>
              <span class="hz-prob"><span class="up">▲ ${pct(h.p_up, 0)}</span> <span class="down">▼ ${pct(h.p_down, 0)}</span></span>
            </button>`).join('')}
        </div>

        <div class="predict-grid">
          <div id="hzSummary" class="hz-summary"></div>
          <div class="chart-panel">
            <div id="chartBox" class="chart-box"></div>
            <div id="chartTip" class="chart-tip" hidden></div>
          </div>
        </div>

        <div class="target-tool">
          <h4>What's the chance of reaching a price?</h4>
          <div class="target-controls">
            <label class="target-input"><span>Target price ₹</span>
              <input id="targetInput" type="number" step="0.05" min="0"></label>
            <input id="targetSlider" type="range" min="-15" max="15" step="0.25" aria-label="Target move in percent">
            <span id="targetPctLabel" class="target-pct"></span>
          </div>
          <div id="targetResult" class="target-result"></div>
        </div>

        <h4>All horizons</h4>
        <div class="table-wrap"><table class="data-table compact hz-table">
          <thead><tr><th>Horizon</th><th>Until</th><th>Likely price</th><th>Change</th><th>90% range</th><th>▲ Up</th><th>▼ Down</th><th id="thAbove">Ends ≥ target</th><th>Touches target</th></tr></thead>
          <tbody id="hzTable"></tbody></table></div>

        <div class="refresh-row">
          <label class="toggle"><input id="autoRefresh" type="checkbox" ${state.auto && d.market_open ? 'checked' : ''} ${d.market_open ? '' : 'disabled'}>
            <span>Auto-refresh every minute${d.market_open ? '' : ' (when the market is open)'}</span></label>
          <button type="button" id="refreshNow" class="btn-secondary">Refresh now</button>
          <span class="muted">Updated ${fmtClock(d.now)} IST</span>
        </div>

        <div id="todayBox" class="today-box"><p class="muted">Checking today’s accuracy…</p></div>

        <details class="coef-details"><summary>How is this predicted?</summary>
          <div class="how">
            <p><b>1 day → 1 month:</b> ${esc(d.model.daily_mean)} expected return + ${esc(d.model.daily_vol)} volatility (each picked by AIC on 5 years of daily prices). Forecast volatility ≈ ${NS.num(d.model.daily_vol_annual_pct, 1)}% a year.</p>
            <p><b>1 & 10 minutes:</b> recent 1-minute volatility (EWMA, ${NS.num(d.model.minute_sigma_pct, 3)}% per minute from ${d.model.minute_obs.toLocaleString('en-IN')} minute moves), Student-t fat tails for 1 minute. When the market is shut the overnight opening-gap risk (±${NS.num(d.model.gap_sigma_pct, 2)}%) is added.</p>
            <p><b>Ranges:</b> 90% of the time the price should end inside the 90% range. <b>Touches target</b> = chance the price hits it at any moment before the horizon, which is always ≥ the chance of ending beyond it.</p>
          </div>
        </details>
        <p class="metric-note">Probabilities from a statistical model, not a promise and not investment advice. Short-term direction is close to a coin flip (≈50%) — the useful part is the size of the likely move. Exchange holidays are not known to the model.</p>
      </article>`;

    view.querySelectorAll('.hz-btn').forEach((b) => b.addEventListener('click', () => { state.horizon = b.dataset.k; updateHorizon(); }));
    const input = $('targetInput'), slider = $('targetSlider');
    const pctNow = state.targetPct[d.ticker] ?? 2;
    slider.value = pctNow;
    input.value = targetPrice(d).toFixed(2);
    slider.addEventListener('input', () => {
      state.targetPct[d.ticker] = parseFloat(slider.value);
      input.value = targetPrice(d).toFixed(2);
      updateTarget();
    });
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      if (!(v > 0)) return;
      state.targetPct[d.ticker] = (v / d.price - 1) * 100;
      slider.value = Math.max(-15, Math.min(15, state.targetPct[d.ticker]));
      updateTarget();
    });
    $('autoRefresh').addEventListener('change', (e) => { state.auto = e.target.checked; e.target.checked ? startRefresh() : stopRefresh(); });
    $('refreshNow').addEventListener('click', () => refresh(true));
    updateHorizon();
    if (state.auto && d.market_open) startRefresh();
    loadToday(d.ticker);
  }

  function updateHorizon() {
    const d = state.data[state.current];
    const h = horizonOf(d);
    state.horizon = h.key;
    view.querySelectorAll('.hz-btn').forEach((b) => {
      const on = b.dataset.k === h.key;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on);
    });
    const up = h.p_up;
    $('hzSummary').innerHTML = `
      <div class="hz-when">Price at <b>${fmtWhen(h.target_time)} IST</b> (${esc(h.label)} ahead)</div>
      <div class="hz-median">${inr(h.median)}</div>
      <div class="hz-change ${tone(h.change_pct)}">${signed(h.change_pct)} expected · typical move ±${h.typical_move_pct.toFixed(2)}%</div>
      <div class="prob-bar" role="img" aria-label="Probability up ${pct(up)}, down ${pct(1 - up)}">
        <span class="pb-up" style="width:${(up * 100).toFixed(1)}%"></span><span class="pb-down" style="width:${((1 - up) * 100).toFixed(1)}%"></span>
      </div>
      <div class="prob-legend"><span class="up">▲ Up ${pct(up)}</span><span class="down">▼ Down ${pct(1 - up)}</span></div>
      <div class="range-rows">
        <div><span>50% likely between</span><b>${inr(h.lo50)} – ${inr(h.hi50)}</b></div>
        <div><span>90% likely between</span><b>${inr(h.lo90)} – ${inr(h.hi90)}</b></div>
      </div>
      ${h.note ? `<p class="hz-note">${esc(h.note)}</p>` : ''}`;
    updateTarget();
  }

  function updateTarget() {
    const d = state.data[state.current];
    const h = horizonOf(d);
    const target = targetPrice(d);
    const tp = state.targetPct[d.ticker] ?? 2;
    $('targetPctLabel').textContent = `${tp >= 0 ? '+' : ''}${tp.toFixed(2)}% vs ${inr(d.price)}`;
    const above = target >= d.price;
    const pEnd = above ? pAbove(h, target) : 1 - pAbove(h, target);
    $('targetResult').innerHTML = `
      <div class="kpi"><span class="k-label">Ends ${above ? 'at or above' : 'at or below'} ${inr(target)} · ${esc(h.label)}</span><span class="k-value">${pct(pEnd)}</span></div>
      <div class="kpi"><span class="k-label">Touches ${inr(target)} at any time before then</span><span class="k-value">${pct(pTouch(h, target))}</span></div>
      <div class="kpi"><span class="k-label">Ends on the other side</span><span class="k-value">${pct(1 - pEnd)}</span></div>`;
    $('thAbove').textContent = above ? 'Ends ≥ target' : 'Ends ≤ target';
    $('hzTable').innerHTML = d.horizons.map((x) => {
      if (x.error) return `<tr><th>${esc(x.label)}</th><td colspan="8" class="muted">${esc(x.error)}</td></tr>`;
      const pe = above ? pAbove(x, target) : 1 - pAbove(x, target);
      return `<tr class="${x.key === h.key ? 'row-best' : ''}" data-k="${x.key}">
        <th>${esc(x.label)}</th><td>${fmtWhen(x.target_time)}</td><td>${inr(x.median)}</td>
        <td class="${tone(x.change_pct)}">${signed(x.change_pct)}</td>
        <td>${inr(x.lo90, 0)} – ${inr(x.hi90, 0)}</td>
        <td class="pos">${pct(x.p_up)}</td><td class="neg">${pct(x.p_down)}</td>
        <td class="hl">${pct(pe)}</td><td>${pct(pTouch(x, target))}</td></tr>`;
    }).join('');
    $('hzTable').querySelectorAll('tr[data-k]').forEach((tr) => tr.addEventListener('click', () => { state.horizon = tr.dataset.k; updateHorizon(); }));
    drawChart(d, h, target);
  }

  // ----------------------------------------------------------------- chart
  // History line + forecast cone (median, 50% and 90% bands) + target line.
  // Built as SVG so it scales, themes with CSS and supports hover.
  function drawChart(d, h, target) {
    const box = $('chartBox'), tip = $('chartTip');
    const minute = h.unit === 'min';
    let hist, cone;
    if (minute) {
      const H = d.history.intraday;
      hist = H.t.slice(-90).map((t, i) => ({ t, p: H.p.slice(-90)[i] }));
      cone = d.paths.minute;
    } else {
      const H = d.history.daily;
      hist = H.t.slice(-60).map((t, i) => ({ t: t + 'T15:30:00+05:30', p: H.p.slice(-60)[i] }));
      cone = d.paths.day.slice(0, Math.max(h.n, 5));
    }
    if (!hist.length || !cone.length) { box.innerHTML = '<p class="muted">Chart data unavailable.</p>'; return; }
    // the last history point is the forecast anchor (live price or last close)
    hist[hist.length - 1] = { ...hist[hist.length - 1], p: d.price };
    const gap = minute && !d.market_open ? 4 : 0;          // visual break for the closed market
    const z90 = 1.6449, z50 = 0.6745;
    const pts = [
      ...hist.map((r) => ({ kind: 'hist', t: r.t, p: r.p })),
      ...cone.map((c) => ({ kind: 'fc', t: c.t, med: d.price * Math.exp(c.mu),
        lo90: d.price * Math.exp(c.mu - z90 * c.sd), hi90: d.price * Math.exp(c.mu + z90 * c.sd),
        lo50: d.price * Math.exp(c.mu - z50 * c.sd), hi50: d.price * Math.exp(c.mu + z50 * c.sd) })),
    ];
    // draw at the box's real pixel width so text stays readable on phones
    const W = Math.max(320, Math.round(box.clientWidth - 16) || 760);
    const Hh = W < 520 ? 250 : 320, L = W < 520 ? 52 : 64, R = 16, T = 16, B = 34;
    // piecewise x-axis: history gets ~62 % of the width, the forecast ~38 %
    // (plus a small break when the market is closed) so the cone is readable
    const HIST_FRAC = 0.62, plotW = W - L - R;
    const gapW = gap ? plotW * 0.05 : 0;
    const histW = plotW * HIST_FRAC, fcW = plotW - histW - gapW;
    const nF = pts.length - hist.length;
    const xAt = (i) => (i < hist.length
      ? L + (hist.length > 1 ? i / (hist.length - 1) : 1) * histW
      : L + histW + gapW + ((i - hist.length + 1) / nF) * fcW);
    const idx = (i) => i;
    const vals = [];
    pts.forEach((q) => (q.kind === 'hist' ? vals.push(q.p) : vals.push(q.lo90, q.hi90)));
    if (target > 0) vals.push(target);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.08 || hi * 0.01;
    lo -= pad; hi += pad;
    const yAt = (v) => T + (1 - (v - lo) / (hi - lo)) * (Hh - T - B);
    const fcs = pts.map((q, i) => ({ q, i })).filter((o) => o.q.kind === 'fc');
    const anchor = { x: xAt(idx(hist.length - 1)), y: yAt(d.price) };
    const band = (k1, k2) => {
      const top = fcs.map((o) => `${xAt(idx(o.i)).toFixed(1)},${yAt(o.q[k2]).toFixed(1)}`);
      const bot = fcs.map((o) => `${xAt(idx(o.i)).toFixed(1)},${yAt(o.q[k1]).toFixed(1)}`).reverse();
      return `${anchor.x.toFixed(1)},${anchor.y.toFixed(1)} ${top.join(' ')} ${bot.join(' ')}`;
    };
    const histLine = pts.filter((q) => q.kind === 'hist').map((q, i) => `${xAt(idx(i)).toFixed(1)},${yAt(q.p).toFixed(1)}`).join(' ');
    const medLine = `${anchor.x.toFixed(1)},${anchor.y.toFixed(1)} ` + fcs.map((o) => `${xAt(idx(o.i)).toFixed(1)},${yAt(o.q.med).toFixed(1)}`).join(' ');
    const ticks = Array.from({ length: 5 }, (_, k) => lo + (k / 4) * (hi - lo));
    const sel = fcs[Math.min(h.n, fcs.length) - 1];
    const xLabels = [0, Math.floor(hist.length / 2), hist.length - 1].map((i) => ({ i, t: pts[i].t, a: i === 0 ? 'start' : 'middle' }))
      .concat(sel ? [{ i: sel.i, t: sel.q.t, a: 'end' }] : []);
    const lab = (t) => (minute ? fmtClock(t) : fmtDay(t));

    box.innerHTML = `
      <svg viewBox="0 0 ${W} ${Hh}" class="pchart" role="img" aria-label="${esc(short(d.ticker))} price history and ${esc(h.label)} forecast range">
        <g class="grid">${ticks.map((v) => `<line x1="${L}" x2="${W - R}" y1="${yAt(v).toFixed(1)}" y2="${yAt(v).toFixed(1)}"/><text x="${L - 8}" y="${(yAt(v) + 4).toFixed(1)}" text-anchor="end">${inr(v, 0)}</text>`).join('')}</g>
        <g class="xlabels">${xLabels.map((o) => `<text x="${xAt(o.i).toFixed(1)}" y="${Hh - 10}" text-anchor="${o.a}">${lab(o.t)}</text>`).join('')}</g>
        ${gap ? `<g class="gap"><line x1="${(L + histW + gapW / 2).toFixed(1)}" x2="${(L + histW + gapW / 2).toFixed(1)}" y1="${T}" y2="${Hh - B}"/><text x="${(L + histW + gapW / 2).toFixed(1)}" y="${T + 12}" text-anchor="middle">closed</text></g>` : ''}
        <polygon class="band90" points="${band('lo90', 'hi90')}"/>
        <polygon class="band50" points="${band('lo50', 'hi50')}"/>
        <polyline class="hist" points="${histLine}"/>
        <polyline class="median" points="${medLine}"/>
        ${target > 0 ? `<line class="target" x1="${L}" x2="${W - R}" y1="${yAt(target).toFixed(1)}" y2="${yAt(target).toFixed(1)}"/><text class="target-lbl" x="${W - R - 4}" y="${(yAt(target) - 6).toFixed(1)}" text-anchor="end">Target ${inr(target, 0)}</text>` : ''}
        ${sel ? `<line class="sel" x1="${xAt(idx(sel.i)).toFixed(1)}" x2="${xAt(idx(sel.i)).toFixed(1)}" y1="${T}" y2="${Hh - B}"/><circle class="sel-dot" cx="${xAt(idx(sel.i)).toFixed(1)}" cy="${yAt(sel.q.med).toFixed(1)}" r="5"/>` : ''}
        <line id="cross" class="cross" x1="0" x2="0" y1="${T}" y2="${Hh - B}" visibility="hidden"/>
        <rect class="hit" x="${L}" y="${T}" width="${W - L - R}" height="${Hh - T - B}"/>
      </svg>
      <div class="chart-legend"><span><i class="lg-hist"></i>${minute ? 'Last session (1-min)' : 'Daily close'}</span><span><i class="lg-med"></i>Likely path</span><span><i class="lg-b50"></i>50% range</span><span><i class="lg-b90"></i>90% range</span><span><i class="lg-tgt"></i>Target</span></div>`;

    const svg = box.querySelector('svg'), cross = box.querySelector('#cross'), hit = box.querySelector('.hit');
    const show = (ev) => {
      const rect = svg.getBoundingClientRect();
      const x = ((ev.clientX - rect.left) / rect.width) * W;
      let best = 0, bd = Infinity;
      pts.forEach((q, i) => { const dd = Math.abs(xAt(idx(i)) - x); if (dd < bd) { bd = dd; best = i; } });
      const q = pts[best], cx = xAt(idx(best));
      cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('visibility', 'visible');
      tip.innerHTML = q.kind === 'hist'
        ? `<b>${minute ? fmtWhen(q.t) : fmtDay(q.t)}</b><br>${best === hist.length - 1 ? (d.market_open ? 'Live' : 'Last close') : 'Price'} ${inr(q.p)}`
        : `<b>${fmtWhen(q.t)}</b><br>Likely ${inr(q.med)}<br>50%: ${inr(q.lo50, 0)} – ${inr(q.hi50, 0)}<br>90%: ${inr(q.lo90, 0)} – ${inr(q.hi90, 0)}`;
      tip.hidden = false;
      const px = (cx / W) * rect.width;
      tip.style.left = `${Math.min(Math.max(px + 12, 0), rect.width - 190)}px`;
      tip.style.top = '12px';
    };
    hit.addEventListener('mousemove', show);
    hit.addEventListener('touchstart', (e) => show(e.touches[0]), { passive: true });
    hit.addEventListener('mouseleave', () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); });
  }

  // ------------------------------------------------- today's accuracy panel
  // Scores the model against the latest session using only data available at
  // the time of each forecast. One day is a small sample, so the panel shows
  // the rolling minute checks (hundreds per day) next to the opening calls.
  const todayCache = {};
  const ok = (b) => (b ? '<span class="pos">✓ Inside</span>' : '<span class="neg">✗ Outside</span>');
  const dirTxt = (b) => (b === null || b === undefined ? '<span class="muted">no move</span>' : b ? '<span class="pos">✓ Right</span>' : '<span class="neg">✗ Wrong</span>');
  const calib = (v, target) => {
    if (!isNum(v)) return '—';
    const gap = v - target;
    const tol = target >= 90 ? 3 : 5;    // 90% range: within ±3 pts; 50% range: ±5 pts
    const verdict = Math.abs(gap) <= tol ? 'on target' : gap < 0 ? 'a bit narrow' : 'a bit wide';
    return `${v.toFixed(1)}% <span class="muted">(${verdict})</span>`;
  };

  async function loadToday(ticker, force) {
    const box = $('todayBox');
    if (!box) return;
    const hit = todayCache[ticker];
    if (hit && !force && Date.now() - hit.at < 60 * 1000) { renderToday(hit.data); return; }
    try {
      const r = await NS.postJSON('/api/predict/accuracy', { symbol: ticker });
      todayCache[ticker] = { at: Date.now(), data: r };
      if (state.current === ticker) renderToday(r);
    } catch (e) {
      if (state.current === ticker && $('todayBox')) $('todayBox').innerHTML = `<p class="card-error">${esc(e.message)}</p>`;
    }
  }

  function renderToday(t) {
    const box = $('todayBox');
    if (!box) return;
    if (t.error) { box.innerHTML = `<h4>How accurate was the model today?</h4><p class="muted">${esc(t.error)}</p>`; return; }
    const day = fmt(t.session + 'T12:00:00+05:30', { weekday: 'short', day: 'numeric', month: 'short' });
    const m1 = t.rolling_1m, m10 = t.rolling_10m;
    const oneDay = t.opening.find((o) => o.label === '1 day');
    const opens = t.opening.map((o) => `
      <tr><th>${esc(o.label)}${o.final ? '' : ' <span class="tag">live</span>'}</th>
        <td>${fmtClock(o.final ? o.target : o.as_of)}</td><td>${inr(o.pred)}</td><td>${inr(o.lo90, 0)} – ${inr(o.hi90, 0)}</td>
        <td>${inr(o.actual)}</td><td>${signed(o.error_pct)}</td><td>${ok(o.in50)}</td><td>${ok(o.in90)}</td><td>${dirTxt(o.direction_ok)}</td></tr>`).join('');
    const roll = (name, m) => (m ? `
      <tr><th>${name}</th><td>${m.n}</td><td>${m.accuracy.toFixed(2)}%</td><td>${NS.num(m.theil_u, 3)}</td>
        <td>${isNum(m.hit_rate) ? m.hit_rate.toFixed(1) + '%' : '—'}</td><td>${calib(m.cover50, 50)}</td><td>${calib(m.cover90, 90)}</td></tr>` : '');
    box.innerHTML = `
      <h4>How accurate was the model today? <span class="muted">· ${day}${t.live ? ` · live, as of ${fmtClock(t.as_of)} IST` : ' · session closed'}</span></h4>
      <div class="kpi-row">
        <div class="kpi"><span class="k-label">1-day forecast accuracy${oneDay && !oneDay.final ? ' (so far)' : ''}</span><span class="k-value">${oneDay ? (100 - Math.abs(oneDay.error_pct)).toFixed(2) + '%' : '—'}</span><span class="k-sub">${oneDay ? `${inr(oneDay.pred)} predicted vs ${inr(oneDay.actual)}` : ''}</span></div>
        <div class="kpi"><span class="k-label">10-min forecasts inside 90% range</span><span class="k-value">${m10 ? m10.cover90.toFixed(0) + '%' : '—'}</span><span class="k-sub">${m10 ? `${m10.n} checks · target 90%` : ''}</span></div>
        <div class="kpi"><span class="k-label">1-min forecasts inside 90% range</span><span class="k-value">${m1 ? m1.cover90.toFixed(0) + '%' : '—'}</span><span class="k-sub">${m1 ? `${m1.n} checks · target 90%` : ''}</span></div>
        <div class="kpi"><span class="k-label">Direction right (1-min / 10-min)</span><span class="k-value">${m1 && isNum(m1.hit_rate) ? m1.hit_rate.toFixed(0) : '—'}% / ${m10 && isNum(m10.hit_rate) ? m10.hit_rate.toFixed(0) : '—'}%</span><span class="k-sub">50% = coin flip</span></div>
      </div>
      <div id="todayChart" class="chart-box"></div>
      <p class="metric-note">Forecasts made from the previous close (${inr(t.prev_close)}, ${esc(t.prev_date)}) — what this page showed before the market opened:</p>
      <div class="table-wrap"><table class="data-table compact">
        <thead><tr><th>Horizon</th><th>Checked at</th><th>Predicted</th><th>90% range</th><th>Actual</th><th>Error</th><th>50% range</th><th>90% range</th><th>Direction</th></tr></thead>
        <tbody>${opens}</tbody></table></div>
      <p class="metric-note">Rolling checks — at every minute (and every 10 minutes) the model predicted ahead using only data up to then:</p>
      <div class="table-wrap"><table class="data-table compact">
        <thead><tr><th>Forecast</th><th>Checks</th><th>Accuracy</th><th>vs naive (Theil’s U)</th><th>Direction right</th><th>Inside 50% range</th><th>Inside 90% range</th></tr></thead>
        <tbody>${roll('1 minute ahead', m1)}${roll('10 minutes ahead', m10)}</tbody></table></div>
      <p class="metric-note">Accuracy = 100 − average % error. It is always high over minutes because prices barely move; Theil’s U ≈ 1.00 means it ties with “the price stays where it is”. The ranges are what to judge: about 50% and 90% of real prices should fall inside them. One day is a small sample.</p>`;
    drawToday(t);
  }

  // Today's price with each 10-minute forecast: dot = predicted, bar = 90% range.
  function drawToday(t) {
    const box = $('todayChart');
    if (!box || !t.chart.t.length) return;
    const W = Math.max(320, Math.round(box.clientWidth - 16) || 760);
    const H = W < 520 ? 230 : 280, L = W < 520 ? 52 : 64, R = 16, T = 14, B = 30;
    const t0 = Date.parse(t.chart.t[0]), t1 = Date.parse(t.chart.t[t.chart.t.length - 1]);
    const xAt = (iso) => L + ((Date.parse(iso) - t0) / Math.max(1, t1 - t0)) * (W - L - R);
    const vals = [...t.chart.p];
    t.chart.f10.forEach((f) => vals.push(f.lo90, f.hi90));
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.06 || hi * 0.005; lo -= pad; hi += pad;
    const yAt = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    const line = t.chart.t.map((iso, i) => `${xAt(iso).toFixed(1)},${yAt(t.chart.p[i]).toFixed(1)}`).join(' ');
    const ticks = Array.from({ length: 4 }, (_, k) => lo + (k / 3) * (hi - lo));
    const marks = t.chart.f10.map((f) => {
      const x = xAt(f.t1).toFixed(1), inside = f.actual >= f.lo90 && f.actual <= f.hi90;
      return `<g class="${inside ? 'fin' : 'fout'}"><line x1="${x}" x2="${x}" y1="${yAt(f.hi90).toFixed(1)}" y2="${yAt(f.lo90).toFixed(1)}"/><circle cx="${x}" cy="${yAt(f.pred).toFixed(1)}" r="3.5"/></g>`;
    }).join('');
    const nOut = t.chart.f10.filter((f) => f.actual < f.lo90 || f.actual > f.hi90).length;
    const labels = [t.chart.t[0], t.chart.t[Math.floor(t.chart.t.length / 2)], t.chart.t[t.chart.t.length - 1]];
    box.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" class="pchart" role="img" aria-label="Today's price with each 10-minute forecast and its 90% range; ${nOut} of ${t.chart.f10.length} fell outside">
        <g class="grid">${ticks.map((v) => `<line x1="${L}" x2="${W - R}" y1="${yAt(v).toFixed(1)}" y2="${yAt(v).toFixed(1)}"/><text x="${L - 8}" y="${(yAt(v) + 4).toFixed(1)}" text-anchor="end">${inr(v, 0)}</text>`).join('')}</g>
        <g class="xlabels">${labels.map((iso, i) => `<text x="${xAt(iso).toFixed(1)}" y="${H - 8}" text-anchor="${i === 0 ? 'start' : i === 2 ? 'end' : 'middle'}">${fmtClock(iso)}</text>`).join('')}</g>
        <polyline class="hist" points="${line}"/>
        <g class="f10">${marks}</g>
        <line id="tcross" class="cross" x1="0" x2="0" y1="${T}" y2="${H - B}" visibility="hidden"/>
        <rect class="hit" x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}"/>
      </svg>
      <div class="chart-legend"><span><i class="lg-hist"></i>Actual price today</span><span><i class="lg-in"></i>10-min forecast, real price inside 90% range</span><span><i class="lg-out"></i>outside (${nOut} of ${t.chart.f10.length})</span></div>
      <div id="todayTip" class="chart-tip" hidden></div>`;
    const svg = box.querySelector('svg'), hitR = box.querySelector('.hit'), tip = $('todayTip'), cross = $('tcross');
    box.style.position = 'relative';
    const show = (ev) => {
      const rect = svg.getBoundingClientRect();
      const x = ((ev.clientX - rect.left) / rect.width) * W;
      let best = null, bd = Infinity;
      t.chart.f10.forEach((f) => { const dd = Math.abs(xAt(f.t1) - x); if (dd < bd) { bd = dd; best = f; } });
      if (!best) return;
      const cx = xAt(best.t1);
      cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('visibility', 'visible');
      const inside = best.actual >= best.lo90 && best.actual <= best.hi90;
      tip.innerHTML = `<b>${fmtClock(best.t0)} → ${fmtClock(best.t1)}</b><br>Predicted ${inr(best.pred)}<br>90%: ${inr(best.lo90)} – ${inr(best.hi90)}<br>Actual ${inr(best.actual)} · ${inside ? '✓ inside' : '✗ outside'}`;
      tip.hidden = false;
      tip.style.left = `${Math.min(Math.max((cx / W) * rect.width + 12, 0), rect.width - 200)}px`;
      tip.style.top = '10px';
    };
    hitR.addEventListener('mousemove', show);
    hitR.addEventListener('touchstart', (e) => show(e.touches[0]), { passive: true });
    hitR.addEventListener('mouseleave', () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); });
  }

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if ($('hzSummary') && state.current && state.data[state.current] && !state.data[state.current].error) updateTarget();
      const tc = todayCache[state.current];
      if (tc && !tc.data.error && $('todayChart')) drawToday(tc.data);
    }, 150);
  });

  // --------------------------------------------------------------- refresh
  async function refresh(manual) {
    const t = state.current;
    if (!t) return;
    const btn = $('refreshNow');
    if (btn) { btn.disabled = true; btn.textContent = 'Refreshing…'; }
    try {
      const r = await NS.postJSON('/api/predict', { symbol: t });
      state.data[t] = r;
      delete todayCache[t];          // re-score today with the newest minutes
      renderTabs();
      if (t === state.current) renderStock();
    } catch (e) {
      if (manual) NS.showError(e.message);
      if (btn) { btn.disabled = false; btn.textContent = 'Refresh now'; }
    }
  }
  function startRefresh() { stopRefresh(); refreshTimer = setInterval(() => refresh(false), REFRESH_MS); }
  function stopRefresh() { if (refreshTimer) clearInterval(refreshTimer); refreshTimer = null; }
})();
