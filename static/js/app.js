/**
 * app.js — LSP frontend logic
 *
 * Responsibilities:
 *   1. Load S&P 500 universe → populate searchable dropdown
 *   2. Load Google Trends → render trends panel (relative-frequency %)
 *   3. On stock selection → fetch metrics + signal → update stock panel
 *   4. Fetch 1-week lookback → render historical signal vs outcome
 *
 * All API calls use fetch() with error handling and loading states.
 */

'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const stockInput    = document.getElementById('stock-input');
const dropdown      = document.getElementById('dropdown');
const inputSpinner  = document.getElementById('input-spinner');
const emptyState    = document.getElementById('empty-state');
const stockData     = document.getElementById('stock-data');
const stockLoading  = document.getElementById('stock-loading');
const trendsList    = document.getElementById('trends-list');
const trendsSkeleton = document.getElementById('trends-skeleton');

// ── State ─────────────────────────────────────────────────────────────────────
let allStocks = [];        // [{ticker, company, sector}]
let focusedIndex = -1;     // keyboard nav in dropdown
let selectedTicker = null;

// ── Number formatters ─────────────────────────────────────────────────────────

/**
 * Format a value based on its type.
 * Returns '—' for null/undefined/NaN.
 */
function fmt(value, type) {
  if (value === null || value === undefined || value === '' || Number.isNaN(value)) return '—';
  const n = parseFloat(value);
  if (!isFinite(n)) return '—';

  switch (type) {
    case 'price':
      return '$' + n.toFixed(2);
    case 'pct':
      // Value is already a decimal (e.g. 0.432) → multiply by 100
      return (n * 100).toFixed(1) + '%';
    case 'pct_raw':
      // Value is already a percentage (e.g. 43.2)
      return n.toFixed(1) + '%';
    case 'large':
      if (n >= 1e12) return '$' + (n / 1e12).toFixed(2) + 'T';
      if (n >= 1e9)  return '$' + (n / 1e9).toFixed(2) + 'B';
      if (n >= 1e6)  return '$' + (n / 1e6).toFixed(2) + 'M';
      return '$' + n.toFixed(0);
    case 'ratio':
      return n.toFixed(2) + 'x';
    case 'multiple':
      return n.toFixed(1) + 'x';
    case 'plain':
      return n.toFixed(2);
    default:
      return String(value);
  }
}

function setText(id, text, cssClass = '') {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className = el.className.replace(/\b(positive|negative|na)\b/g, '').trim();
  if (cssClass) el.classList.add(cssClass);
  if (text === '—') el.classList.add('na');
}

// ── Signal explanation builder ────────────────────────────────────────────────

function joinList(parts) {
  if (parts.length === 1) return parts[0];
  if (parts.length === 2) return `${parts[0]} and ${parts[1]}`;
  return `${parts.slice(0, -1).join(', ')}, and ${parts[parts.length - 1]}`;
}

function buildExplanation(data) {
  const votes   = data.votes   || {};
  const details = data.details || {};
  const signal  = (data.signal || 'HOLD').toUpperCase();
  const score   = data.score ?? 0;
  const ticker  = data.ticker || 'This stock';
  const sentences = [];

  // No intraday data case
  if (details.error === 'no_intraday_data') {
    return `No intraday data is available for ${ticker} today. The signal defaults to HOLD until the 9:40 AM ET morning refresh provides first-10-minute bar data. Check back after market open.`;
  }

  // Sentence 1 — overall signal and conviction level
  const scoreAbs   = Math.abs(score);
  const conviction = scoreAbs >= 4 ? 'strong' : scoreAbs >= 3 ? 'moderate' : 'marginal';
  if (signal === 'BUY') {
    sentences.push(`${ticker} earns a ${conviction} BUY signal (+${score}/5) based on early-session price action.`);
  } else if (signal === 'SELL') {
    sentences.push(`${ticker} earns a ${conviction} SELL signal (${score}/5) based on early-session price action.`);
  } else {
    sentences.push(`${ticker} earns a HOLD signal (${score > 0 ? '+' : ''}${score}/5), reflecting no strong directional bias in the first 10 minutes of trading.`);
  }

  // Sentences 2-3 — supporting evidence
  const bullParts = [];
  const bearParts = [];

  if (votes.gap === 1 && details.gap_pct != null)
    bullParts.push(`a bullish opening gap of +${Math.abs(details.gap_pct).toFixed(1)}% vs yesterday's close`);
  else if (votes.gap === -1 && details.gap_pct != null)
    bearParts.push(`a bearish opening gap of −${Math.abs(details.gap_pct).toFixed(1)}% vs yesterday's close`);

  if (votes.momentum === 1 && details.momentum_pct != null)
    bullParts.push(`upward 10-min price momentum of +${Math.abs(details.momentum_pct).toFixed(2)}%`);
  else if (votes.momentum === -1 && details.momentum_pct != null)
    bearParts.push(`downward 10-min price momentum of −${Math.abs(details.momentum_pct).toFixed(2)}%`);

  if (votes.vwap === 1)  bullParts.push('price trading above VWAP (institutional buy zone)');
  else if (votes.vwap === -1) bearParts.push('price trading below VWAP (institutional sell zone)');

  if (votes.volume === 1 && details.vol_ratio != null)
    bullParts.push(`elevated volume at ${details.vol_ratio.toFixed(1)}× expected, confirming momentum`);

  if (votes.macro_trend === 1)  bullParts.push('supportive macro Google Trends sentiment');
  else if (votes.macro_trend === -1) bearParts.push('elevated bearish macro search terms (recession/unemployment)');

  if (bullParts.length > 0)
    sentences.push(`Bullish factors: ${joinList(bullParts)}.`);
  if (bearParts.length > 0)
    sentences.push(`Bearish factors: ${joinList(bearParts)}.`);

  // Final sentence — always present
  if (signal === 'BUY')
    sentences.push('These early signals suggest institutional buying pressure — validate with sector context and broader market conditions before acting.');
  else if (signal === 'SELL')
    sentences.push('Early weakness may reflect distribution or news-driven selling — confirm with stop-loss discipline and fundamental checks before acting.');
  else
    sentences.push('Monitor for a volume-confirmed breakout above or below VWAP before committing capital; mixed signals call for patience.');

  return sentences.join(' ');
}

// ── Universe loading ──────────────────────────────────────────────────────────

async function loadUniverse() {
  try {
    inputSpinner.classList.remove('hidden');
    const resp = await fetch('/api/universe');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    allStocks = await resp.json();
    stockInput.placeholder = `Search ${allStocks.length} S&P 500 stocks by ticker or name…`;
  } catch (err) {
    console.warn('Universe load failed:', err);
    stockInput.placeholder = 'Search S&P 500 stocks…';
  } finally {
    inputSpinner.classList.add('hidden');
  }
}

// ── Dropdown logic ────────────────────────────────────────────────────────────

function filterStocks(query) {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  return allStocks
    .filter(s =>
      s.ticker.toLowerCase().startsWith(q) ||
      s.company.toLowerCase().includes(q)
    )
    .slice(0, 25); // cap at 25 for performance
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
    item.addEventListener('mousedown', e => {
      e.preventDefault();
      selectStock(item.dataset.ticker);
    });
  });
}

function closeDropdown() {
  dropdown.innerHTML = '';
  focusedIndex = -1;
}

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
    const focused = items[focusedIndex];
    if (focused) selectStock(focused.dataset.ticker);
  } else if (e.key === 'Escape') {
    closeDropdown();
    stockInput.blur();
  }
});

stockInput.addEventListener('blur', () => {
  // Small delay so mousedown on dropdown item fires first
  setTimeout(closeDropdown, 150);
});

// ── Stock selection ───────────────────────────────────────────────────────────

async function selectStock(ticker) {
  selectedTicker = ticker;
  const stock = allStocks.find(s => s.ticker === ticker);

  // Update input to show selected ticker
  stockInput.value = stock ? `${stock.ticker} — ${stock.company}` : ticker;
  closeDropdown();

  // Show loading state
  emptyState.classList.add('hidden');
  stockData.classList.add('hidden');
  stockLoading.classList.remove('hidden');

  try {
    const resp = await fetch(`/api/stock/${encodeURIComponent(ticker)}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    renderStockPanel(data);

    // Start loading history in the background (non-blocking)
    loadHistory(ticker);
  } catch (err) {
    console.error('Stock fetch failed:', err);
    stockLoading.classList.add('hidden');
    emptyState.classList.remove('hidden');
    document.querySelector('.empty-title').textContent = 'Failed to load data';
    document.querySelector('.empty-sub').textContent = 'Please try again in a moment';
  }
}

// ── Stock panel rendering ─────────────────────────────────────────────────────

function renderStockPanel(data) {
  // Company header
  document.getElementById('company-name').textContent =
    data.company_name || data.ticker;
  document.getElementById('ticker-badge').textContent = data.ticker;
  document.getElementById('sector-tag').textContent   = data.sector || '';

  // Price
  const price = data.current_price;
  document.getElementById('current-price').textContent =
    price != null ? fmt(price, 'price') : '—';

  // Signal badge
  const badge = document.getElementById('signal-badge');
  const signal = (data.signal || 'HOLD').toUpperCase();
  badge.textContent = signal;
  badge.className = 'signal-badge';
  badge.classList.add(
    signal === 'BUY'  ? 'signal-buy'  :
    signal === 'SELL' ? 'signal-sell' :
    signal === 'HOLD' ? 'signal-hold' : 'signal-none'
  );

  // Score bar (range -5 to +5, normalised to 0-100%)
  const score = data.score ?? 0;
  const pct   = Math.round(((score + 5) / 10) * 100);
  const fill  = document.getElementById('score-fill');
  fill.style.width = pct + '%';
  fill.style.background =
    signal === 'BUY'  ? '#00C853' :
    signal === 'SELL' ? '#FF1744' : '#FFB300';
  document.getElementById('score-value').textContent = `${score > 0 ? '+' : ''}${score} / 5`;

  // Financial metrics
  setText('m-pe',          fmt(data.pe_ratio,       'plain'));
  setText('m-eps',         data.eps != null ? fmt(data.eps, 'price') : '—');
  setText('m-mktcap',      fmt(data.market_cap,     'large'));
  setText('m-revenue',     fmt(data.revenue,        'large'));
  setText('m-grossmargin', fmt(data.gross_margin,   'pct'));
  setText('m-de',          fmt(data.debt_to_equity, 'ratio'));
  setText('m-roe',         fmt(data.roe,            'pct'));
  setText('m-pb',          fmt(data.pb_ratio,       'multiple'));

  // 52-week range with position indicator
  const hi = data.week_52_high, lo = data.week_52_low, cp = data.current_price;
  if (hi != null && lo != null) {
    document.getElementById('m-52w').textContent =
      `$${parseFloat(lo).toFixed(2)} – $${parseFloat(hi).toFixed(2)}`;
    const rangeFill  = document.getElementById('m-52w-fill');
    const rangeLabel = document.getElementById('m-52w-label');
    if (rangeFill && cp != null && hi > lo) {
      const pct = Math.min(100, Math.max(0, ((cp - lo) / (hi - lo)) * 100));
      rangeFill.style.width = pct.toFixed(1) + '%';
      if (rangeLabel) rangeLabel.textContent = `▸ ${pct.toFixed(0)}% of range`;
    }
  } else {
    document.getElementById('m-52w').textContent = '—';
    document.getElementById('m-52w').classList.add('na');
  }

  const dy = data.dividend_yield;
  document.getElementById('m-divyield').textContent =
    dy != null ? (dy * 100).toFixed(2) + '%' : '—';
  if (dy == null) document.getElementById('m-divyield').classList.add('na');

  // Apply colour hints to key metrics
  applyMetricColors(data);

  // Signal explanation
  const explanationEl = document.getElementById('signal-explanation');
  if (explanationEl) explanationEl.textContent = buildExplanation(data);

  // Indicator votes
  renderVotes(data.votes || {});

  // Reset history section to loading spinner
  document.getElementById('history-list').innerHTML =
    '<div class="history-loading"><div class="spinner-ring"></div></div>';

  // Show panel
  stockLoading.classList.add('hidden');
  stockData.classList.remove('hidden');
}

function applyMetricColors(data) {
  // ROE: green if > 15%, red if negative
  if (data.roe != null) {
    const el = document.getElementById('m-roe');
    if (data.roe > 0.15)  el.classList.add('positive');
    else if (data.roe < 0) el.classList.add('negative');
  }
  // Gross margin: green if > 40%, red if < 10%
  if (data.gross_margin != null) {
    const el = document.getElementById('m-grossmargin');
    if (data.gross_margin > 0.40)  el.classList.add('positive');
    else if (data.gross_margin < 0.10) el.classList.add('negative');
  }
  // D/E: red if > 2x
  if (data.debt_to_equity != null) {
    const el = document.getElementById('m-de');
    if (data.debt_to_equity > 200) el.classList.add('negative');
  }
}

function renderVotes(votes) {
  const container = document.getElementById('votes-row');
  const labels = {
    gap:          'Gap',
    momentum:     'Momentum',
    vwap:         'VWAP',
    volume:       'Volume',
    macro_trend:  'Macro',
  };

  container.innerHTML = Object.entries(votes).map(([key, val]) => {
    const label = labels[key] || key;
    const cls   = val > 0 ? 'vote-up' : val < 0 ? 'vote-down' : 'vote-flat';
    const icon  = val > 0 ? '↑' : val < 0 ? '↓' : '→';
    return `<span class="vote-chip ${cls}">${icon} ${label}</span>`;
  }).join('');
}

// ── 1-Week Lookback ───────────────────────────────────────────────────────────

async function loadHistory(ticker) {
  const container = document.getElementById('history-list');
  try {
    const resp = await fetch(`/api/stock/${encodeURIComponent(ticker)}/history`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    // Guard against stale responses if the user switched stocks mid-load
    if (ticker !== selectedTicker) return;
    renderHistory(data.history || []);
  } catch (err) {
    console.warn('History fetch failed:', err);
    if (ticker !== selectedTicker) return;
    container.innerHTML = '<p class="history-empty">Historical data unavailable</p>';
  }
}

function renderHistory(history) {
  const container = document.getElementById('history-list');

  if (!history.length) {
    container.innerHTML = '<p class="history-empty">No completed trading days available yet</p>';
    return;
  }

  // Column header row
  const headerHtml = `
    <div class="history-header">
      <span class="hist-date">Date</span>
      <span class="hist-signal-hdr">Signal</span>
      <span class="hist-return-hdr">Stock Move</span>
      <span class="hist-result-hdr">Your Gain if Followed</span>
    </div>
  `;

  // Data rows
  const rowsHtml = history.map(day => {
    // Parse date as local noon to avoid timezone-shift issues
    const dateObj = new Date(day.date + 'T12:00:00');
    const dateStr = dateObj.toLocaleDateString('en-US', {
      weekday: 'short', month: 'short', day: 'numeric'
    });

    const signalCls =
      day.signal === 'BUY'  ? 'signal-buy'  :
      day.signal === 'SELL' ? 'signal-sell' : 'signal-hold';

    const retSign = day.day_return_pct >= 0 ? '+' : '';
    const retCls  = day.day_return_pct >= 0 ? 'positive' : 'negative';

    let resultHtml;
    if (day.profitable === true) {
      resultHtml = `<span class="hist-result win">✓ +${day.signal_return_pct.toFixed(2)}%</span>`;
    } else if (day.profitable === false) {
      resultHtml = `<span class="hist-result loss">✗ ${day.signal_return_pct.toFixed(2)}%</span>`;
    } else {
      resultHtml = `<span class="hist-result hold">— Held (no trade)</span>`;
    }

    return `
      <div class="history-item">
        <span class="hist-date">${dateStr}</span>
        <span class="hist-signal ${signalCls}">${day.signal}</span>
        <span class="hist-return ${retCls}">${retSign}${day.day_return_pct.toFixed(2)}%</span>
        ${resultHtml}
      </div>
    `;
  }).join('');

  // Summary row — total gain from following all signals (HOLD days = 0%)
  const totalGain = history.reduce((sum, d) => sum + (d.signal_return_pct ?? 0), 0);
  const totalSign = totalGain >= 0 ? '+' : '';
  const totalCls  = totalGain > 0 ? 'win' : totalGain < 0 ? 'loss' : 'hold';
  const tradedDays = history.filter(d => d.signal !== 'HOLD').length;
  const summaryHtml = `
    <div class="history-summary">
      <span class="history-summary-label">5-Day Total Return (following all signals)</span>
      <span class="hist-result ${totalCls}" style="margin-left:0">${totalSign}${totalGain.toFixed(2)}% across ${tradedDays} trade${tradedDays !== 1 ? 's' : ''}</span>
    </div>
  `;

  container.innerHTML = headerHtml + rowsHtml + summaryHtml;
}

// ── Google Trends panel ───────────────────────────────────────────────────────

async function loadTrends() {
  try {
    const resp = await fetch('/api/trends');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const payload = await resp.json();
    renderTrends(payload.data || []);
  } catch (err) {
    console.warn('Trends load failed:', err);
    renderTrends([]);
  }
}

function renderTrends(trends) {
  if (!trends.length) {
    trendsList.innerHTML = '<p style="color:#476085;font-size:11px;padding:8px 0;">Trends unavailable</p>';
    return;
  }

  // Compute total score for relative-frequency display
  const totalScore = trends.reduce((sum, t) => sum + t.score, 0) || 1;
  // Keep max-score for bar proportions (so bars fill the space visually)
  const maxScore = Math.max(...trends.map(t => t.score), 1);

  const html = trends.map((t, i) => {
    const relPct  = Math.round((t.score / totalScore) * 100); // share of total
    const barPct  = Math.round((t.score / maxScore) * 100);   // visual bar width
    const barClass = barPct >= 70 ? 'high' : barPct >= 40 ? 'medium' : 'low';

    // HOT = relative share ≥ 20 %; NOT = share ≤ 5 %
    const badgeCls = relPct >= 20 ? 'hot' : relPct <= 5 ? 'not' : 'warm';
    const badgeTxt = relPct >= 20 ? 'HOT' : relPct <= 5 ? 'NOT' : '';

    return `
      <div class="trend-item">
        <span class="trend-rank">${i + 1}</span>
        <span class="trend-term">${t.term}</span>
        <span class="trend-badge ${badgeCls}">${badgeTxt}</span>
        <div class="trend-bar-wrap">
          <div class="trend-bar ${barClass}" style="width:0%" data-target="${barPct}%"></div>
        </div>
        <span class="trend-score">${relPct}%</span>
      </div>
    `;
  }).join('');

  trendsList.innerHTML = html;

  // Animate bars in after DOM insertion
  requestAnimationFrame(() => {
    trendsList.querySelectorAll('.trend-bar').forEach(bar => {
      bar.style.width = bar.dataset.target;
    });
  });

  // Macro vote
  updateMacroVote(trends, totalScore);
}

function updateMacroVote(trends, totalScore) {
  const voteEl = document.getElementById('macro-vote');
  if (!voteEl) return;

  const total = totalScore || trends.reduce((s, t) => s + t.score, 0) || 1;

  const BULLISH = new Set(['stock market', 'S&P 500', 'earnings report']);
  const BEARISH = new Set(['recession', 'unemployment']);

  // Use relative shares (%) for the vote calculation
  const relShares = Object.fromEntries(
    trends.map(t => [t.term, (t.score / total) * 100])
  );

  const avgShare = 100 / Math.max(trends.length, 1); // ~10% for 10 terms
  const bullAvg = [...BULLISH].reduce((s, t) => s + (relShares[t] ?? avgShare), 0) / BULLISH.size;
  const bearAvg = [...BEARISH].reduce((s, t) => s + (relShares[t] ?? avgShare), 0) / BEARISH.size;
  const diff = (bullAvg - bearAvg) / 100; // normalise to [-1, +1]

  let label, cls;
  if (diff > 0.05)       { label = 'Bullish';  cls = 'bullish'; }
  else if (diff < -0.05) { label = 'Bearish';  cls = 'bearish'; }
  else                   { label = 'Neutral';  cls = 'neutral'; }

  voteEl.textContent = label;
  voteEl.className = `macro-vote ${cls}`;
}

// ── Init ──────────────────────────────────────────────────────────────────────

(async function init() {
  // Load universe and trends in parallel
  await Promise.all([loadUniverse(), loadTrends()]);
})();
