"""
backtest.py  (v2 — daily-data signal replay)
----------------------------------------------
Replays the *exact same* BUY/HOLD/SELL signal logic as the live intraday
model (src/fetcher.py) on historical daily OHLCV bars.

Why daily bars instead of 1-minute bars?
-----------------------------------------
yfinance free tier provides 1-minute intraday data for only the trailing
7 calendar days.  Historical 5-year minute data is not available.  Daily
bars go back 10+ years and are fully reliable.

Daily-bar proxies for each live-model indicator
------------------------------------------------
  Gap       : (Open_t  - Close_{t-1}) / Close_{t-1} × 100
              Same threshold ±1.0 % as the live model.

  Momentum  : (Close_t - Open_t) / Open_t × 100
              Same threshold ±0.3 % as the live model.
              Uses open-to-close return as the "first 10-minute" proxy.

  VWAP      : Close_t vs (High_t + Low_t + Close_t) / 3  (typical price)
              Same threshold ±0.1 % as the live model.

  Volume    : Volume_t / rolling_20-day avg volume (no lookahead)
              Same threshold 1.5× as the live model.
              High volume confirms the momentum direction.

  Macro     : Neutral (0) — cannot reconstruct historical macro vote.

Scoring / signal thresholds
-----------------------------
  score = sum of 4 votes  (range -4 to +4, same as live model minus macro)
  BUY  if score ≥ +2
  SELL if score ≤ -2
  HOLD otherwise

Forward return definition
--------------------------
  fwd = Close_{t+1} / Close_t − 1   (next trading-day close-to-close)

For a BUY signal on day t we ask: did the stock go up the next day?
For a SELL signal on day t: did it go down?
This mirrors the spirit of the live model (act on the morning signal,
measure what happened by the next close).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# ── Signal thresholds — must match src/fetcher.py ─────────────────────────────
_GAP_PCT_THRESHOLD    = 1.0    # %
_MOMENTUM_PCT_THRESHOLD = 0.3  # %
_VWAP_PCT_THRESHOLD   = 0.1    # %
_VOL_HIGH_RATIO       = 1.5
_VOL_WINDOW           = 20     # days for rolling volume average
_BUY_THRESHOLD        = 2
_SELL_THRESHOLD       = -2

# ── Fetch periods by window ────────────────────────────────────────────────────
# Each entry: (yfinance period to download, target trading-day count)
_WINDOW_CONFIG: Dict[str, tuple] = {
    "1mo": ("6mo",  21),
    "1y":  ("2y",  252),
    "5y":  ("10y", 1260),
}


# ── Per-bar signal computation ─────────────────────────────────────────────────

def _compute_daily_signal(
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    prev_close: float,
    avg_volume: Optional[float],
) -> tuple[str, int, dict]:
    """Return (signal, score, votes) for one daily bar."""
    votes: dict[str, int] = {}

    # 1. Gap
    if prev_close > 0:
        gap_pct = (open_ - prev_close) / prev_close * 100.0
        votes["gap"] = 1 if gap_pct >= _GAP_PCT_THRESHOLD else (
            -1 if gap_pct <= -_GAP_PCT_THRESHOLD else 0
        )
    else:
        votes["gap"] = 0

    # 2. Momentum (open-to-close)
    if open_ > 0:
        mom_pct = (close - open_) / open_ * 100.0
        votes["momentum"] = 1 if mom_pct >= _MOMENTUM_PCT_THRESHOLD else (
            -1 if mom_pct <= -_MOMENTUM_PCT_THRESHOLD else 0
        )
    else:
        votes["momentum"] = 0

    # 3. VWAP proxy: close vs typical price
    typical = (high + low + close) / 3.0
    if typical > 0:
        vs_vwap = (close - typical) / typical * 100.0
        votes["vwap"] = 1 if vs_vwap >= _VWAP_PCT_THRESHOLD else (
            -1 if vs_vwap <= -_VWAP_PCT_THRESHOLD else 0
        )
    else:
        votes["vwap"] = 0

    # 4. Volume confirms momentum direction
    if avg_volume and avg_volume > 0:
        vol_ratio = volume / avg_volume
        votes["volume"] = votes["momentum"] if vol_ratio >= _VOL_HIGH_RATIO else 0
    else:
        votes["volume"] = 0

    score = sum(votes.values())
    signal = "BUY" if score >= _BUY_THRESHOLD else (
        "SELL" if score <= _SELL_THRESHOLD else "HOLD"
    )
    return signal, score, votes


# ── Summary stats ──────────────────────────────────────────────────────────────

def _empty_summary() -> Dict:
    return {
        "buy_avg_fwd_return":  0.0,
        "hold_avg_fwd_return": 0.0,
        "sell_avg_fwd_return": 0.0,
        "buy_minus_sell":      0.0,
        "accuracy":            0.0,
        "n_buy": 0, "n_hold": 0, "n_sell": 0,
        "cumulative_strategy_return": 0.0,
        "cumulative_buy_and_hold":    0.0,
    }


def _summarize(rows: List[Dict], full_close: pd.Series) -> Dict:
    if not rows:
        return _empty_summary()

    df = pd.DataFrame(rows)
    df["fwd_return"] = pd.to_numeric(df["fwd_return"], errors="coerce")

    def mean_of(mask) -> float:
        sub = df.loc[mask, "fwd_return"].dropna()
        return float(sub.mean()) if len(sub) else 0.0

    buy_mask  = df["signal"] == "BUY"
    hold_mask = df["signal"] == "HOLD"
    sell_mask = df["signal"] == "SELL"

    buy_avg  = mean_of(buy_mask)
    hold_avg = mean_of(hold_mask)
    sell_avg = mean_of(sell_mask)

    # Accuracy: BUYs where fwd > 0  + SELLs where fwd < 0, averaged
    buy_hits  = df.loc[buy_mask,  "fwd_return"].dropna().gt(0).mean() if buy_mask.any()  else 0.0
    sell_hits = df.loc[sell_mask, "fwd_return"].dropna().lt(0).mean() if sell_mask.any() else 0.0
    if buy_mask.any() and sell_mask.any():
        accuracy = float((buy_hits + sell_hits) / 2.0)
    elif buy_mask.any():
        accuracy = float(buy_hits)
    elif sell_mask.any():
        accuracy = float(sell_hits)
    else:
        accuracy = 0.0

    # Cumulative strategy: long on BUY, short on SELL, flat on HOLD
    legs = []
    for _, row in df.iterrows():
        fwd = row["fwd_return"]
        if fwd is None or (isinstance(fwd, float) and math.isnan(fwd)):
            continue
        if row["signal"] == "BUY":
            legs.append(fwd)
        elif row["signal"] == "SELL":
            legs.append(-fwd)
        else:
            legs.append(0.0)
    strat_cum = float(np.prod([1 + x for x in legs]) - 1) if legs else 0.0

    # Buy-and-hold over the actual window rows
    bh = float(full_close.iloc[-1] / full_close.iloc[0] - 1) if len(full_close) >= 2 else 0.0

    return {
        "buy_avg_fwd_return":  round(buy_avg,  6),
        "hold_avg_fwd_return": round(hold_avg, 6),
        "sell_avg_fwd_return": round(sell_avg, 6),
        "buy_minus_sell":      round(buy_avg - sell_avg, 6),
        "accuracy":            round(accuracy, 4),
        "n_buy":  int(buy_mask.sum()),
        "n_hold": int(hold_mask.sum()),
        "n_sell": int(sell_mask.sum()),
        "cumulative_strategy_return": round(strat_cum, 6),
        "cumulative_buy_and_hold":    round(bh, 6),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def run_backtest(ticker: str, window: str = "1y") -> Dict:
    """
    Replay the live signal model on `window` of daily history for `ticker`.

    Returns a dict with:
      window, ticker, rows, summary  — detailed results
      hit_rate, buy_sell_spread, avg_buy_return, avg_sell_return  — flat summary
        for direct consumption by app.py and the frontend.
    """
    if window not in _WINDOW_CONFIG:
        raise ValueError(f"window must be one of {list(_WINDOW_CONFIG)}")

    fetch_period, target_days = _WINDOW_CONFIG[window]
    ticker = ticker.upper()

    # Download daily OHLCV — extra history so volume average has warmup
    try:
        hist = yf.Ticker(ticker).history(
            period=fetch_period, interval="1d", auto_adjust=True
        )
    except Exception as exc:
        log.warning("backtest(%s, %s): history fetch failed: %s", ticker, window, exc)
        return _empty_result(ticker, window)

    if hist is None or hist.empty:
        return _empty_result(ticker, window)

    hist = hist[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if len(hist) < _VOL_WINDOW + 2:
        return _empty_result(ticker, window)

    # Rolling 20-day average volume, shifted by 1 to avoid lookahead
    avg_vol_series = hist["Volume"].rolling(_VOL_WINDOW).mean().shift(1)

    # Trim to the target window for signal generation (keep extra rows for warmup)
    window_hist = hist.iloc[-target_days - 1:]   # +1 so day 0 provides prev_close

    rows: List[Dict] = []
    for i in range(1, len(window_hist)):
        today     = window_hist.iloc[i]
        yesterday = window_hist.iloc[i - 1]
        date_idx  = window_hist.index[i]

        avg_vol_val = avg_vol_series.get(date_idx)
        avg_vol = float(avg_vol_val) if (avg_vol_val is not None and not math.isnan(float(avg_vol_val))) else None

        signal, score, votes = _compute_daily_signal(
            open_      = float(today["Open"]),
            high       = float(today["High"]),
            low        = float(today["Low"]),
            close      = float(today["Close"]),
            volume     = float(today["Volume"]),
            prev_close = float(yesterday["Close"]),
            avg_volume = avg_vol,
        )

        # Forward return: next trading day's close vs today's close
        if i + 1 < len(window_hist):
            fwd = float(window_hist.iloc[i + 1]["Close"] / today["Close"] - 1)
        else:
            fwd = None

        rows.append({
            "date":       date_idx.strftime("%Y-%m-%d"),
            "price":      round(float(today["Close"]), 4),
            "signal":     signal,
            "score":      score,
            "votes":      votes,
            "fwd_return": round(fwd, 6) if fwd is not None else None,
        })

    window_closes = window_hist["Close"].iloc[1:]   # exclude the warmup row
    summary = _summarize(rows, window_closes)

    return {
        "window":  window,
        "ticker":  ticker,
        "rows":    rows,
        "summary": summary,
        # Flat convenience keys consumed by app.py and app.js
        "hit_rate":        summary["accuracy"],
        "buy_sell_spread": summary["buy_minus_sell"],
        "avg_buy_return":  summary["buy_avg_fwd_return"],
        "avg_sell_return": summary["sell_avg_fwd_return"],
    }


def _empty_result(ticker: str, window: str) -> Dict:
    s = _empty_summary()
    return {
        "window":  window,
        "ticker":  ticker,
        "rows":    [],
        "summary": s,
        "hit_rate":        s["accuracy"],
        "buy_sell_spread": s["buy_minus_sell"],
        "avg_buy_return":  s["buy_avg_fwd_return"],
        "avg_sell_return": s["sell_avg_fwd_return"],
    }


def run_multi_window_backtest(ticker: str) -> Dict[str, Dict]:
    """Run 1mo / 1y / 5y backtests in one call."""
    out: Dict[str, Dict] = {}
    for w in ("1mo", "1y", "5y"):
        try:
            out[w] = run_backtest(ticker, window=w)
        except Exception as exc:
            log.warning("backtest(%s, %s) failed: %s", ticker, w, exc)
            out[w] = _empty_result(ticker, w)
            out[w]["error"] = str(exc)
    return out
