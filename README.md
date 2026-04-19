# LSP — Longshot Stock Prediction

**MBA 6223 Final Project · The Ohio State Unviversity**
**Author: R.S. Dunn**

> An educational equity signal dashboard that applies a nine-factor scoring model to S&P 500 stocks, combining intraday price action, technical regime filters, macro sentiment, and Wall Street analyst consensus into a single daily BUY / HOLD / SELL verdict.

**Live demo:** [lsp-stock-prediction.onrender.com](https://lsp-stock-prediction.onrender.com)

---

## What This Is

LSP is a single-page Flask web application that ingests live market data each morning and scores every S&P 500 stock across nine indicators. The result is a ranked, explainable signal — not a black box — displayed alongside fundamentals, historical hit rate, and a full backtest so users can judge model quality for themselves.

The tool is built for educational purposes: to explore how quantitative signals can be constructed, validated, and iterated on using freely available data.

---

## Dashboard Overview

### Search & Signal Header

Type any S&P 500 company name or ticker into the search bar. The model fetches live intraday data and returns a verdict within seconds.

Each stock displays:

- **Current price** and today's change ($ and %)
- **Verdict badge** — BUY / HOLD / SELL with conviction label (strong / moderate / marginal / mixed)
- **Score bar** — visual position on a −11 to +11 scale showing where the stock sits today
- **"Why this signal"** — plain-English explanation of which factors drove the verdict
- **Vote chips** — one chip per indicator showing its direction (↑ bullish · ↓ bearish · → neutral)
- **Track record** — model hit rate across the last 30 scored days and 5-day cumulative return if followed

### Four Tab Panels

| Tab | Contents |
|---|---|
| **Signal** | Full narrative explanation · 52-week price range with position indicator |
| **Fundamentals** | Nine trailing metrics (P/E, EPS, market cap, revenue, gross margin, D/E, ROE, P/B, dividend yield) · Five forward estimates (Forward P/E, Forward EPS, price target with % upside, consensus rating, analyst coverage count) |
| **Backtest** | 1-year hit rate, BUY−SELL spread, avg BUY/SELL return · Signal-vs-outcome scatter chart |
| **History** | Last 5 completed trading days: signal given, actual stock move, cumulative return if followed |

### Embedded Dashboard Sections

Scrolling below the main panel reveals two embedded tools:

**Backtest** — Enter any ticker and run 1-month, 1-year, or 5-year backtests. Displays strategy return vs. buy-and-hold with interactive Plotly charts across all three windows simultaneously. Automatically populates for whichever stock is selected in the search bar.

**Alpha Ranking** — Ranks the full S&P 500 by 12-month return relative to the index, showing top outperformers and bottom underperformers in a side-by-side table. Loads on page open.

---

## Signal Model

### Nine Indicators (Score Range: −11 to +11)

Each indicator votes +1 (bullish), −1 (bearish), or 0 (neutral). The analyst indicator votes up to ±3 (triple-weighted per optimizer). Votes sum to a raw score.

| # | Indicator | Source | Vote Logic |
|---|---|---|---|
| 1 | **Gap** | Yahoo Finance intraday | ±1 if opening gap ≥ ±0.5% vs prior close |
| 2 | **Momentum** | Yahoo Finance intraday | ±1 if 10-min return ≥ ±0.3% |
| 3 | **VWAP** | Yahoo Finance intraday | ±1 if last price deviates ≥ ±0.3% from VWAP |
| 4 | **Volume** | Yahoo Finance intraday | ±1 if first-10-min volume ≥ 2.0× 20-day avg (mirrors momentum direction) |
| 5 | **Macro trend** | Google Trends | ±1 from macro search proxy (passed in from trends module) |
| 6 | **RSI-14** | Yahoo Finance daily | +1 if RSI > 50 · −1 if RSI < 50 (Wilder EWM, 3-month lookback) |
| 7 | **MA-50** | Yahoo Finance daily | +1 if price above 50-day MA · −1 if below |
| 8 | **Sector ETF** | Yahoo Finance daily | ±1 by same-day return of sector ETF (XLK, XLF, XLV, etc.) |
| 9 | **Analyst consensus** | Yahoo Finance (FactSet/Refinitiv) | **+3** if `recommendationMean` ≤ 2.5 · **−3** if ≥ 3.5 · 0 otherwise |

**Signal rules:**

| Score | Verdict | Position |
|---|---|---|
| ≥ +1 | **BUY** | Long (full position) |
| −1 to +1 | **HOLD** | Flat (cash) |
| ≤ −1 | **SELL** | Flat (cash) — long-only, no shorts |

**VIX gate:** BUY suppressed to HOLD when VIX ≥ 25, regardless of score — a high-fear market override.

---

## Threshold Optimization

Thresholds were determined by a grid-search optimizer (`optimize_thresholds.py`) across 25 diversified S&P 500 stocks using one year of daily OHLCV data, measuring cumulative strategy return minus buy-and-hold alpha.

### Optimizer v4 Results — 15,360 combinations tested

**Analyst weight sensitivity** (most significant finding):

| Analyst weight | Avg alpha vs B&H | vs. disabled |
|---|---|---|
| 0× (disabled) | −36.8% | — |
| 1× | −32.5% | +4.3 ppt |
| 2× | −28.7% | +8.1 ppt |
| **3× (selected)** | **−25.7%** | **+11.1 ppt** |

Analyst weight improves alpha monotonically. The model selects **analyst_w = 3**.

**Note:** 24 of 25 sample tickers are currently rated Buy by analysts — the weight acts more as a quality tilt toward stocks the street endorses than as a stock picker. It has the most discriminating power in environments with diverging analyst opinion.

**Optimized thresholds (v4):**

```
Gap threshold:       ±0.5%    (higher bar — analyst weight anchors direction)
Momentum threshold:  ±0.3%
VWAP threshold:      ±0.3%
Volume ratio:        2.0×     (requires strong volume confirmation)
Buy threshold:       +1
VIX gate:            25.0
Analyst weight:      3×
```

---

## Recent Improvements

### Analyst Consensus Integration (latest)
- Added `recommendationMean`, `forwardPE`, `forwardEps`, `targetMeanPrice`, `targetHigh/LowPrice`, and `numberOfAnalystOpinions` from Yahoo Finance (FactSet/Refinitiv consensus) as both a signal vote and a new Fundamentals display
- Analyst vote is **triple-weighted** per optimizer — reduces alpha gap by 11 percentage points vs. disabled
- Price target shown with % upside/downside to current price, color-coded green/red
- Source attribution note displayed in the Fundamentals tab: data origin, update cadence, and weight rationale

### Score Bar (±11 visual scale)
- Replaced numeric-only score with an animated bar spanning −11 to +11
- Color-coded fill (green for BUY, red for SELL) grows from center toward the indicator dot
- Verdict badge shows score and conviction together: e.g., `+8 / 11 · strong BUY`
- Backtest chart and score axis match the same scale

### Technical Indicators Added (RSI, MA-50, Sector ETF, VIX gate)
- **RSI-14:** computed from 3-month daily bars using Wilder's EWM smoothing; > 50 = bullish regime
- **50-day MA:** trend regime filter; price above/below MA contributes direction
- **Sector ETF:** daily return of the stock's GICS-sector ETF as confirmation; uses a full sector map (XLK, XLF, XLV, XLY, XLC, XLP, XLI, XLE, XLB, XLRE, XLU)
- **VIX gate:** suppresses BUY → HOLD when VIX ≥ 25 — forward-looking risk management retained even where the 2024 backtest showed no benefit

### Long-Only Strategy
- SELL signals now result in flat (cash) rather than short positions
- Improved average alpha by ~12 percentage points over the shorting version across the 2024 bull-market sample period

### Embedded Backtest & Alpha Ranking
- Both tools integrated into the main dashboard page (no separate nav required)
- Selecting a stock from the search bar automatically runs the backtest section
- Three chart windows displayed side-by-side: 1 month, 1 year, 5 years

---

## Architecture

```
app.py                      Flask routes + APScheduler (9:40 AM ET batch refresh)
src/
  fetcher.py                Signal computation, yfinance data fetching, analyst data
  universe.py               S&P 500 ticker list (Wikipedia)
  trends.py                 Google Trends macro proxy
finance_depth/
  backtest.py               Historical signal replay on daily OHLCV bars
  alpha_ranker.py           12-month alpha vs. S&P 500
optimize_thresholds.py      Grid-search threshold optimizer (v4, 15,360 combos)
templates/index.html        Single-page dashboard
static/js/app.js            Frontend logic (fetch, render, Plotly charts)
static/css/style.css        Editorial design system (warm paper + ink palette)
```

---

## Data Sources

| Data | Source | Refresh cadence |
|---|---|---|
| Intraday OHLCV (1-min bars) | Yahoo Finance via `yfinance` | Daily at 9:40 AM ET |
| Daily OHLCV (3-month, for RSI/MA50) | Yahoo Finance via `yfinance` | Daily |
| Analyst estimates & consensus | Yahoo Finance (FactSet/Refinitiv) | Per stock selection + daily batch |
| VIX level | Yahoo Finance (`^VIX`) | Daily |
| Sector ETF returns | Yahoo Finance (XLK, XLF, XLV, etc.) | Daily |
| Macro sentiment | Google Trends (market proxy terms) | Daily |
| S&P 500 universe | Wikipedia | On startup |

---

## Run Locally

```bash
git clone https://github.com/hartmanh14/MBA6223_OSU_Final_Project
cd MBA6223_OSU_Final_Project

python -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
python app.py
```

Open [http://localhost:5000](http://localhost:5000). First load takes 10–20 seconds while live data is fetched and cached.

---

## How the Daily Refresh Works

1. **APScheduler (in-process):** fires `_daily_refresh()` at 9:40 AM ET, Monday–Friday
2. **External cron via `/api/refresh`:** POST to this endpoint from [cron-job.org](https://cron-job.org) at 9:42 AM ET to handle Render cold starts on the free tier
3. **Lazy fallback:** if any user requests data after 9:40 AM with no cache, a background refresh is triggered automatically

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `REFRESH_SECRET` | Recommended | Protects `/api/refresh`. Auto-generated by Render. |
| `LOG_LEVEL` | Optional | `INFO` (default) or `DEBUG` |
| `PORT` | Set by Render | Do not set manually |

---

## Disclaimers

**Educational use only.** This tool is not investment advice and should not be used to make financial decisions.

- Signals are generated from publicly available data with inherent delays and gaps
- The backtest uses daily OHLCV proxies for intraday indicators — an approximation of live model behavior
- Analyst consensus uses today's rating as a static feature across the historical backtest window (ratings are sticky but not unchanged)
- Past model performance does not predict future results
