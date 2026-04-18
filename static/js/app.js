/**
 * app.js — LSP frontend logic (v2, editorial redesign)
 *
 *   1. Load S&P 500 universe → searchable dropdown
 *   2. Load Google Trends → collapsed drawer, one-line macro verdict inline
 *   3. On stock selection → render hero (verdict + why + votes + track record),
 *      plus four tab panels (Signal / Fundamentals / Backtest / History)
 *   4. Fetch 1-week lookback → populate History panel
 */

'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const stockInput    = document.getElementById('stock-input');
const dropdown      = document.getElementById('dropdown');
const inputSpinner  = document.getElementById('input-spinner');
const emptyState    = document.getElementById('empty-state');
const stockData     = document.getElementById('stock-data');
const stockLoading  = document.getElementById('stock-loading');
const tabsEl        = document.getElementById('tabs');
const panelsEls     = document.querySelectorAll('[data-panel]');
const trendsDrawer  = document.getElementById('trends-drawer');
const trendsList    = document.getElementById('trends-list');
const trendsToggle  = document.getElementById('trends-toggle');
const trendsClose   = document.getElementById('trends-close');

// ── State ─────────────────────────────────────────────────────────────────────
let allStocks = [];
let focusedIndex = -1;
let selectedTicker = null;
let trendsCache = [];

// ── Formatters ────────────────────────────────────────────────────────────────
function fmt(value, type) {
  if (value === null || value === undefined || value === '' || Number.isNaN(value)) return '—';
  const n = parseFloat(value);
  if (!isFinite(n)) return '—';

  switch (type) {
    case 'price':  return '$' + n.toFixed(2);
    case 'pct':    return (n * 100).toFixed(1) + '%';
    case 'pct_raw': return n.toFixed(1) + '%';
    case 'large':
      if (n >= 1e12) return '$' + (n / 1e12).toFixed(2) + 'T';
      if (n >= 1e9)  return '$' + (n / 1e9).toFixed(2) + 'B';
      if (n >= 1e6)  return '$' + (n / 1e6).toFixed(2) + 'M';
      return '$' + n.toFixed(0);
    case 'ratio':    return n.toFixed(2) + 'x';
    case 'multiple': return n.toFixed(1) + 'x';
    case 'plain':    return n.toFixed(2);
    default:         return String(value);
  }
}

function setText(id, text, cssClass = '') {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.classList.remove('up', 'down', 'na');
  if (cssClass) el.classList.add(cssClass);
  if (text === '—') el.classList.add('na');
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
tabsEl.addEventListener('click', (e) => {
  const tab = e.target.closest('.tab');
  if (!tab) return;
  const key = tab.dataset.tab;
  tabsEl.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t === tab));
  panelsEls.forEach(p => p.classList.toggle('hidden', p.dataset.panel !== key));
});

// ── Trends drawer toggle ──────────────────────────────────────────────────────
trendsToggle?.addEventListener('click', () => {
  trendsDrawer.classList.remove('hidden');
  trendsToggle.textContent = 'hide details ↑';
});
trendsClose?.addEventListener('click', () => {
  trendsDrawer.classList.add('hidden');
  trendsToggle.textContent = 'show 10 terms →';
});

// ── Signal explanation (concise version for the WHY card) ────────────────────
function buildWhy(data) {
  const v = data.votes || {};
  const d = data.details || {};
  const signal = (data.signal || 'HOLD').toUpperCase();

  if (d.error === 'no_intraday_data') {
    return 'No intraday data yet today — the signal defaults to HOLD until the 9:40 AM ET refresh.';
  }

  const parts = [];
  if (v.gap === 1  && d.gap_pct != null) parts.push(`opening gap of <strong>+${Math.abs(d.gap_pct).toFixed(1)}%</strong>`);
  if (v.gap === -1 && d.gap_pct != null) parts.push(`opening gap of <strong>−${Math.abs(d.gap_pct).toFixed(1)}%</strong>`);
  if (v.momentum === 1  && d.momentum_pct != null) parts.push(`upward 10-min momentum of <strong>+${Math.abs(d.momentum_pct).toFixed(2)}%</strong>`);
  if (v.momentum === -1 && d.momentum_pct != null) parts.push(`downward 10-min momentum of <strong>−${Math.abs(d.momentum_pct).toFixed(2)}%</strong>`);
  if (v.vwap === 1)  parts.push('price <strong>above VWAP</strong>');
  if (v.vwap === -1) parts.push('price <strong>below VWAP</strong>');
  if (v.volume === 1 && d.vol_ratio != null) parts.push(`volume at <strong>${d.vol_ratio.toFixed(1)}×</strong> expected`);
  if (v.macro_trend === 1)  parts.push('supportive macro sentiment');
  if (v.macro_trend === -1) parts.push('bearish macro sentiment');

  if (!parts.length) {
    return signal === 'HOLD'
      ? 'Mixed signals across gap, momentum, VWAP, and volume — no directional edge today.'
      : 'Evidence details unavailable.';
  }
  return parts.join(', ') + '.';
}

// Narrative — the longer version for the Signal tab
function buildNarrative(data) {
  const votes   = data.votes   || {};
  const details = data.details || {};
  const signal  = (data.signal || 'HOLD').toUpperCase();
  const score   = data.score ?? 0;
  const ticker  = data.ticker || 'This stock';
  const sentences = [];

  if (details.error === 'no_intraday_data') {
    return `No intraday data is available for ${ticker} today. The signal defaults to HOLD until the 9:40 AM ET morning refresh provides first-10-minute bar data.`;
  }

  const scoreAbs = Math.abs(score);
  const conviction = scoreAbs >= 4 ? 'strong' : scoreAbs >= 3 ? 'moderate' : 'marginal';
  if (signal === 'BUY')
    sentences.push(`${ticker} earns a ${conviction} BUY signal (+${score}/5) based on early-session price action.`);
  else if (signal === 'SELL')
    sentences.push(`${ticker} earns a ${conviction} SELL signal (${score}/5) based on early-session price action.`);
  else
    sentences.push(`${ticker} earns a HOLD signal (${score > 0 ? '+' : ''}${score}/5), reflecting no strong directional bias in the first 10 minutes of trading.`);

  const bull = [], bear = [];
  if (votes.gap === 1  && details.gap_pct != null) bull.push(`a bullish opening gap of +${Math.abs(details.gap_pct).toFixed(1)}%`);
  if (votes.gap === -1 && details.gap_pct != null) bear.push(`a bearish opening gap of −${Math.abs(details.gap_pct).toFixed(1)}%`);
  if (votes.momentum === 1  && details.momentum_pct != null) bull.push(`upward 10-min momentum of +${Math.abs(details.momentum_pct).toFixed(2)}%`);
  if (votes.momentum === -1 && details.momentum_pct != null) bear.push(`downward 10-min momentum of −${Math.abs(details.momentum_pct).toFixed(2)}%`);
  if (votes.vwap === 1)  bull.push('price trading above VWAP (institutional buy zone)');
  if (votes.vwap === -1) bear.push('price trading below VWAP (institutional sell zone)');
  if (votes.volume === 1 && details.vol_ratio != null) bull.push(`elevated volume at ${details.vol_ratio.toFixed(1)}× expected`);
  if (votes.macro_trend === 1)  bull.push('supportive macro sentiment');
  if (votes.macro_trend === -1) bear.push('elevated bearish macro search terms');

  const join = p => p.length === 1 ? p[0] : p.length === 2 ? `${p[0]} and ${p[1]}` : `${p.slice(0,-1).join(', ')}, and ${p[p.length-1]}`;
  if (bull.length) sentences.push(`Bullish factors: ${join(bull)}.`);
  if (bear.length) sentences.push(`Bearish factors: ${join(bear)}.`);

  if (signal === 'BUY')
    sentences.push('These early signals suggest institutional buying pressure — validate with sector context and broader market conditions before acting.');
  else if (signal === 'SELL')
    sentences.push('Early weakness may reflect distribution or news-driven selling — confirm with stop-loss discipline before acting.');
  else
    sentences.push('Monitor for a volume-confirmed breakout above or below VWAP before committing capital; mixed signals call for patience.');

  return sentences.join(' ');
}

// ── Universe ──────────────────────────────────────────────────────────────────
async function loadUniverse() {
  try {
    inputSpinner.classList.remove('hidden');
    const resp = await fetch('/api/universe');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    allStocks = await resp.json();
    stockInput.placeholder = `Search ${allStocks.length} S&P 500 companies…`;
  } catch (err) {
    console.warn('Universe load failed:', err);
  } finally {
    inputSpinner.classList.add('hidden');
  }
}

// ── Dropdown ──────────────────────────────────────────────────────────────────
function filterStocks(q) {
  q = q.trim().toLowerCase();
  if (!q) return [];
  return allStocks
    .filter(s => s.ticker.toLowerCase().startsWith(q) || s.company.toLowerCase().includes(q))
    .slice(0, 25);
}

function renderDropdown(matches) {
  focusedIndex = -1;
  if (!matches.length) {
    dropdown.innerHTML = '<div class="dropdown-no-results">No results</div>';
    return;
  }
  dropdown.innerHTML = matches.map((s, i) => `
    <div class="dropdown-item" data-ticker="${s.ticker}" data-index="${i}">
      <span class="dropdown-ticker">${s.ticker}</span>
      <span class="dropdown-company">${s.company}</span>
      <span class="dropdown-sector">${s.sector || ''}</span>
    </div>
  `).join('');
  dropdown.querySelectorAll('.dropdown-item').forEach(item => {
    item.addEventListener('mousedown', e => { e.preventDefault(); selectStock(item.dataset.ticker); });
  });
}

function closeDropdown() { dropdown.innerHTML = ''; focusedIndex = -1; }

stockInput.addEventListener('input', () => {
  const q = stockInput.value;
  if (!q.trim()) { closeDropdown(); return; }
  renderDropdown(filterStocks(q));
});

stockInput.addEventListener('keydown', e => {
  const items = dropdown.querySelectorAll('.dropdown-item');
  if (!items.length) return;
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    focusedIndex = Math.min(focusedIndex + 1, items.length - 1);
    items.forEach((el, i) => el.classList.toggle('focused', i === focusedIndex));
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    focusedIndex = Math.max(focusedIndex - 1, 0);
    items.forEach((el, i) => el.classList.toggle('focused', i === focusedIndex));
  } else if (e.key === 'Enter' && focusedIndex >= 0) {
    e.preventDefault();
    selectStock(items[focusedIndex].dataset.ticker);
  } else if (e.key === 'Escape') {
    closeDropdown();
    stockInput.blur();
  }
});

stockInput.addEventListener('blur', () => setTimeout(closeDropdown, 150));

// ── Stock selection ───────────────────────────────────────────────────────────
async function selectStock(ticker) {
  selectedTicker = ticker;
  const stock = allStocks.find(s => s.ticker === ticker);
  stockInput.value = stock ? `${stock.ticker} — ${stock.company}` : ticker;
  closeDropdown();

  emptyState.classList.add('hidden');
  stockData.classList.add('hidden');
  stockLoading.classList.remove('hidden');

  try {
    const resp = await fetch(`/api/stock/${encodeURIComponent(ticker)}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    renderStockPanel(data);
    loadHistory(ticker);
  } catch (err) {
    console.error('Stock fetch failed:', err);
    stockLoading.classList.add('hidden');
    emptyState.classList.remove('hidden');
  }
}

// ── Render hero + panels ──────────────────────────────────────────────────────
function renderStockPanel(data) {
  // Identity
  document.getElementById('company-name').textContent = data.company_name || data.ticker;
  const tickerRow = [data.ticker, data.sector].filter(Boolean).join(' · ');
  document.getElementById('ticker-row').textContent = tickerRow;

  // Price + change
  const price = data.current_price;
  document.getElementById('current-price').textContent = price != null ? fmt(price, 'price') : '—';

  const changeEl = document.getElementById('price-change');
  const prev = data.prev_close ?? (data.details && data.details.prev_close);
  if (price != null && prev != null && prev > 0) {
    const delta = price - prev;
    const pct = (delta / prev) * 100;
    const sign = delta >= 0 ? '+' : '−';
    changeEl.textContent = `${sign}$${Math.abs(delta).toFixed(2)} · ${sign}${Math.abs(pct).toFixed(2)}% today`;
    changeEl.classList.toggle('up', delta >= 0);
    changeEl.classList.toggle('down', delta < 0);
  } else {
    changeEl.textContent = '';
  }

  // Verdict badge
  const badge = document.getElementById('verdict-badge');
  const signal = (data.signal || 'HOLD').toUpperCase();
  badge.className = 'verdict-badge ' + (signal === 'BUY' ? 'buy' : signal === 'SELL' ? 'sell' : 'hold');
  document.getElementById('verdict-text').textContent = signal;
  const score = data.score ?? 0;
  const conviction = Math.abs(score) >= 4 ? 'strong' : Math.abs(score) >= 3 ? 'moderate' : 'marginal';
  document.getElementById('verdict-score').textContent =
    `${score > 0 ? '+' : ''}${score} / 5 · ${signal === 'HOLD' ? 'mixed' : conviction}`;

  // Why
  document.getElementById('why-text').innerHTML = buildWhy(data);

  // Votes
  renderVotes(data.votes || {});

  // Track record (inline stats)
  setText('tr-5day', '—');
  document.getElementById('tr-5day-sub').textContent = 'Computing…';
  setText('tr-hit', data.hit_rate != null ? (data.hit_rate * 100).toFixed(0) + '%' : '—');
  document.getElementById('tr-hit-sub').textContent = data.hit_rate != null
    ? 'Across last 30 scored days'
    : 'Insufficient backtest data';

  // Fundamentals panel
  setText('m-pe',          fmt(data.pe_ratio,       'plain'));
  setText('m-eps',         data.eps != null ? fmt(data.eps, 'price') : '—');
  setText('m-mktcap',      fmt(data.market_cap,     'large'));
  setText('m-revenue',     fmt(data.revenue,        'large'));
  setText('m-grossmargin', fmt(data.gross_margin,   'pct'));
  setText('m-de',          fmt(data.debt_to_equity, 'ratio'));
  setText('m-roe',         fmt(data.roe,            'pct'));
  setText('m-pb',          fmt(data.pb_ratio,       'multiple'));
  const dy = data.dividend_yield;
  setText('m-divyield', dy != null ? (dy * 100).toFixed(2) + '%' : '—');
  applyMetricColors(data);

  // 52-week range (Signal tab)
  const hi = data.week_52_high, lo = data.week_52_low, cp = data.current_price;
  if (hi != null && lo != null && cp != null && hi > lo) {
    const pct = Math.min(100, Math.max(0, ((cp - lo) / (hi - lo)) * 100));
    document.getElementById('range-headline').textContent = `${pct.toFixed(0)}% of range`;
    document.getElementById('range-fill').style.width = pct.toFixed(1) + '%';
    document.getElementById('range-lo').textContent = `$${parseFloat(lo).toFixed(2)}`;
    document.getElementById('range-hi').textContent = `$${parseFloat(hi).toFixed(2)}`;
    document.getElementById('range-caption').textContent =
      pct > 80 ? 'Trading near the 52-week high — watch for resistance.' :
      pct < 20 ? 'Trading near the 52-week low — watch for support.' :
      'Price sits in the middle of the 52-week range.';
  } else {
    document.getElementById('range-headline').textContent = '—';
    document.getElementById('range-caption').textContent = 'Range data unavailable.';
  }

  // Narrative (Signal tab)
  document.getElementById('narrative-text').textContent = buildNarrative(data);

  // Backtest tab — populate inline stats
  loadBacktestStats(data.ticker);

  // Reset history panel
  document.getElementById('history-list').innerHTML =
    '<div class="history-loading"><div class="spinner-ring"></div></div>';

  // Reset tabs to Signal
  tabsEl.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === 'signal'));
  panelsEls.forEach(p => p.classList.toggle('hidden', p.dataset.panel !== 'signal'));

  stockLoading.classList.add('hidden');
  stockData.classList.remove('hidden');
}

function applyMetricColors(data) {
  if (data.roe != null) {
    const el = document.getElementById('m-roe');
    if (data.roe > 0.15)  el.classList.add('up');
    else if (data.roe < 0) el.classList.add('down');
  }
  if (data.gross_margin != null) {
    const el = document.getElementById('m-grossmargin');
    if (data.gross_margin > 0.40)  el.classList.add('up');
    else if (data.gross_margin < 0.10) el.classList.add('down');
  }
  if (data.debt_to_equity != null) {
    const el = document.getElementById('m-de');
    if (data.debt_to_equity > 200) el.classList.add('down');
  }
}

function renderVotes(votes) {
  const container = document.getElementById('votes-row');
  const labels = { gap: 'Gap', momentum: 'Momentum', vwap: 'VWAP', volume: 'Volume', macro_trend: 'Macro' };
  container.innerHTML = Object.entries(votes).map(([key, val]) => {
    const label = labels[key] || key;
    const cls = val > 0 ? 'vote-up' : val < 0 ? 'vote-down' : 'vote-flat';
    const icon = val > 0 ? '↑' : val < 0 ? '↓' : '→';
    return `<span class="vote-chip ${cls}">${icon} ${label}</span>`;
  }).join('');
}

// ── Backtest inline stats ─────────────────────────────────────────────────────
async function loadBacktestStats(ticker) {
  const btHitEl     = document.getElementById('bt-hit');
  const btSpreadEl  = document.getElementById('bt-spread');
  const btBuyEl     = document.getElementById('bt-buy-avg');
  const btSellEl    = document.getElementById('bt-sell-avg');
  const btLinkEl    = document.getElementById('bt-link');

  if (btLinkEl) btLinkEl.href = `/backtest?ticker=${encodeURIComponent(ticker)}`;

  try {
    const resp = await fetch(`/api/backtest/${encodeURIComponent(ticker)}/1y`);
    if (!resp.ok) return;
    const bt = await resp.json();

    if (btHitEl && bt.hit_rate != null) {
      btHitEl.textContent = (bt.hit_rate * 100).toFixed(0) + '%';
      btHitEl.classList.toggle('up', bt.hit_rate > 0.5);
      btHitEl.classList.toggle('down', bt.hit_rate < 0.4);
    }
    if (btSpreadEl && bt.buy_sell_spread != null) {
      const s = bt.buy_sell_spread * 100;
      btSpreadEl.textContent = (s >= 0 ? '+' : '') + s.toFixed(1) + '%';
      btSpreadEl.classList.toggle('up', s > 0);
      btSpreadEl.classList.toggle('down', s < 0);
    }
    if (btBuyEl && bt.avg_buy_return != null) {
      const v = bt.avg_buy_return * 100;
      btBuyEl.textContent = (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
      btBuyEl.classList.toggle('up', v > 0);
      btBuyEl.classList.toggle('down', v < 0);
    }
    if (btSellEl && bt.avg_sell_return != null) {
      const v = bt.avg_sell_return * 100;
      btSellEl.textContent = (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
      btSellEl.classList.toggle('up', v < 0);
      btSellEl.classList.toggle('down', v > 0);
    }
  } catch (err) {
    console.warn('Backtest stats failed:', err);
  }
}

// ── History ───────────────────────────────────────────────────────────────────
async function loadHistory(ticker) {
  const container = document.getElementById('history-list');
  try {
    const resp = await fetch(`/api/stock/${encodeURIComponent(ticker)}/history`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    if (ticker !== selectedTicker) return;
    renderHistory(data.history || []);
  } catch (err) {
    console.warn('History fetch failed:', err);
    if (ticker !== selectedTicker) return;
    container.innerHTML = '<p class="history-empty">Historical data unavailable</p>';
    setText('tr-5day', '—');
    document.getElementById('tr-5day-sub').textContent = 'History unavailable';
  }
}

function renderHistory(history) {
  const container = document.getElementById('history-list');

  if (!history.length) {
    container.innerHTML = '<p class="history-empty">No completed trading days available yet</p>';
    setText('tr-5day', '—');
    document.getElementById('tr-5day-sub').textContent = 'No completed days';
    return;
  }

  const headerHtml = `
    <div class="history-header">
      <span class="hist-date">Date</span>
      <span class="hist-signal-hdr">Signal</span>
      <span class="hist-return-hdr">Stock move</span>
      <span class="hist-result-hdr">Cumul. if followed</span>
    </div>`;

  let cumulative = 0;
  const rowsHtml = history.map(day => {
    cumulative += (day.signal_return_pct ?? 0);
    const dateObj = new Date(day.date + 'T12:00:00');
    const dateStr = dateObj.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
    const signalCls = day.signal === 'BUY' ? 'signal-buy' : day.signal === 'SELL' ? 'signal-sell' : 'signal-hold';
    const retSign = day.day_return_pct >= 0 ? '+' : '';
    const retCls  = day.day_return_pct >= 0 ? 'positive' : 'negative';
    const cumSign = cumulative >= 0 ? '+' : '';
    const cumCls  = cumulative > 0 ? 'win' : cumulative < 0 ? 'loss' : 'hold';
    return `
      <div class="history-item">
        <span class="hist-date">${dateStr}</span>
        <span class="hist-signal ${signalCls}">${day.signal}</span>
        <span class="hist-return ${retCls}">${retSign}${day.day_return_pct.toFixed(2)}%</span>
        <span class="hist-result ${cumCls}">${cumSign}${cumulative.toFixed(2)}%</span>
      </div>`;
  }).join('');

  const totalGain = history.reduce((s, d) => s + (d.signal_return_pct ?? 0), 0);
  const totalSign = totalGain >= 0 ? '+' : '';
  const totalCls  = totalGain > 0 ? 'win' : totalGain < 0 ? 'loss' : 'hold';
  const tradedDays = history.filter(d => d.signal !== 'HOLD').length;
  const summaryHtml = `
    <div class="history-summary">
      <span class="history-summary-label">5-day total · following all signals</span>
      <span class="hist-result ${totalCls}">${totalSign}${totalGain.toFixed(2)}%</span>
    </div>`;
  container.innerHTML = headerHtml + rowsHtml + summaryHtml;

  // Feed the hero's "5-day if followed" stat
  setText('tr-5day', `${totalSign}${totalGain.toFixed(2)}%`, totalGain > 0 ? 'up' : totalGain < 0 ? 'down' : '');
  document.getElementById('tr-5day-sub').textContent =
    `Across ${tradedDays} traded day${tradedDays !== 1 ? 's' : ''}, ${history.length - tradedDays} HOLD`;
}

// ── Trends ────────────────────────────────────────────────────────────────────
async function loadTrends() {
  try {
    const resp = await fetch('/api/trends');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const payload = await resp.json();
    trendsCache = payload.data || [];
    renderTrends(trendsCache);
  } catch (err) {
    console.warn('Trends load failed:', err);
    renderTrends([]);
  }
}

function renderTrends(trends) {
  const macroEl = document.getElementById('tr-macro');
  const termsEl = document.getElementById('tr-macro-terms');

  if (!trends.length) {
    if (trendsList) trendsList.innerHTML = '<p class="history-empty">Trends unavailable</p>';
    if (macroEl) macroEl.textContent = '—';
    if (termsEl) termsEl.textContent = '';
    return;
  }

  const totalScore = trends.reduce((s, t) => s + t.score, 0) || 1;
  const maxScore = Math.max(...trends.map(t => t.score), 1);

  if (trendsList) {
    trendsList.innerHTML = trends.map((t, i) => {
      const relPct = Math.round((t.score / totalScore) * 100);
      const barPct = Math.round((t.score / maxScore) * 100);
      return `
        <div class="trend-item">
          <span class="trend-rank">${i + 1}</span>
          <span class="trend-term">${t.term}</span>
          <div class="trend-bar-wrap"><div class="trend-bar" style="width:0%" data-target="${barPct}%"></div></div>
          <span class="trend-score">${relPct}%</span>
        </div>`;
    }).join('');

    requestAnimationFrame(() => {
      trendsList.querySelectorAll('.trend-bar').forEach(b => b.style.width = b.dataset.target);
    });
  }

  // Macro verdict inline
  const BULLISH = new Set(['stock market', 'S&P 500', 'earnings report']);
  const BEARISH = new Set(['recession', 'unemployment']);
  const shares = Object.fromEntries(trends.map(t => [t.term, (t.score / totalScore) * 100]));
  const avgShare = 100 / Math.max(trends.length, 1);
  const bull = [...BULLISH].reduce((s, t) => s + (shares[t] ?? avgShare), 0) / BULLISH.size;
  const bear = [...BEARISH].reduce((s, t) => s + (shares[t] ?? avgShare), 0) / BEARISH.size;
  const diff = (bull - bear) / 100;

  let label, cls;
  if (diff > 0.05)       { label = 'Bullish'; cls = 'up'; }
  else if (diff < -0.05) { label = 'Bearish'; cls = 'down'; }
  else                   { label = 'Neutral'; cls = ''; }

  if (macroEl) {
    macroEl.textContent = label;
    macroEl.classList.remove('up', 'down');
    if (cls) macroEl.classList.add(cls);
  }

  // Inline term summary
  if (termsEl) {
    const top3 = trends.slice(0, 3).map(t => t.term).join(', ');
    termsEl.textContent = `${top3} dominating · `;
  }
}

// ── Refresh ───────────────────────────────────────────────────────────────────
// Clears all server-side caches instantly, then re-fetches trends and the
// currently displayed stock so the UI shows live data immediately.
window.triggerRefresh = async function triggerRefresh() {
  const btn = document.getElementById('refresh-btn');
  btn.disabled = true;
  btn.textContent = 'Clearing…';

  try {
    await fetch('/api/refresh', { method: 'POST' });
  } catch (e) {
    btn.textContent = 'Error';
    setTimeout(() => { btn.disabled = false; btn.textContent = 'Refresh'; }, 3000);
    return;
  }

  btn.textContent = 'Fetching…';
  // Reload trends and the currently displayed stock with fresh live data
  await loadTrends();
  if (selectedTicker) await selectStock(selectedTicker);

  btn.textContent = 'Done!';
  setTimeout(() => { btn.disabled = false; btn.textContent = 'Refresh'; }, 3000);
};

// ── Init ──────────────────────────────────────────────────────────────────────
(async function init() {
  await Promise.all([loadUniverse(), loadTrends()]);
})();
