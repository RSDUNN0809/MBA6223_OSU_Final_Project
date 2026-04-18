"""
backtest.py
-----------
Proof-of-concept backtester for the composite-score Buy/Hold/Sell signal.

What we test
------------
For each bar in the chosen historical window we:

    1. Reconstruct a *point-in-time* price-history snapshot (so RSI at time t
       uses only data available at time t).
    2. Use current fundamentals (P/E, beta, FCF, etc.) as a proxy for their
       historical values — yfinance's free tier doesn't give us a clean
       fundamentals panel, so we hold valuation/DCF constant over the window.
       This is a known, documented limitation; the RSI and price momentum
       components remain fully historical.
    3. Compute the composite score and Buy/Hold/Sell signal for that bar.
    4. Compare the signal to the forward return between that bar and the
       next checkpoint (next day for 1mo, next week for 1y/5y).

Outputs
-------
    {
        "window":  "1mo" | "1y" | "5y",
        "rows":    [ {date, price, composite, signal, fwd_return}, ... ],
        "summary": {
            "buy_avg_fwd_return":  float,
            "hold_avg_fwd_return": float,
            "sell_avg_fwd_return": float,
            "buy_minus_sell":      float,   # headline spread
            "accuracy":            float,   # fraction of BUYs with positive fwd return
            "n_buy":  int, "n_hold": int, "n_sell": int,
            "cumulative_strategy_return": float,
            "cumulative_buy_and_hold":    float,
        }
    }

Usage
-----
    from finance_depth.backtest import run_backtest
    result_1m = run_backtest("AAPL", window="1mo")
    result_1y = run_backtest("AAPL", window="1y")
    result_5y = run_backtest("AAPL", window="5y")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .fundamentals import (
    Fundamentals,
    compute_fundamentals,
    compute_rsi,
    score_rsi,
    score_eps_surprise,
    score_valuation,
)
from .composite import compute_composite_score, DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD

log = logging.getLogger(__name__)


WINDOW_TO_YFINANCE = {
    "1mo": ("3mo", "1d", 1),   # fetch 3mo history so 14-day RSI has warmup, step=1d
    "1y":  ("15mo", "1d", 5),  # weekly sampling for 1y
    "5y":  ("6y", "1d", 21),   # roughly monthly sampling for 5y
}


@dataclass
class BacktestRow:
    date: str
    price: float
    rsi: Optional[float]
    composite: float
    signal: str
    fwd_return: Optional[float]

    def to_dict(self) -> Dict:
        return asdict(self)


def _period_for(window: str) -> (str, str, int):
    if window not in WINDOW_TO_YFINANCE:
        raise ValueError(f"window must be one of {list(WINDOW_TO_YFINANCE)}")
    return WINDOW_TO_YFINANCE[window]


def _iter_checkpoints(closes: pd.Series, step: int, lookback_needed: int = 15):
    """Yield (checkpoint_idx, checkpoint_date) for each sampled bar."""
    n = len(closes)
    start = max(lookback_needed, 0)
    for i in range(start, n, step):
        yield i, closes.index[i]


def _trim_to_window(closes: pd.Series, window: str) -> pd.Series:
    """Keep only the bars within the requested user-facing window."""
    if closes.empty:
        return closes
    end = closes.index[-1]
    if window == "1mo":
        start = end - pd.Timedelta(days=32)
    elif window == "1y":
        start = end - pd.Timedelta(days=370)
    else:  # 5y
        start = end - pd.Timedelta(days=365 * 5 + 10)
    return closes[closes.index >= start]


def run_backtest(
    ticker: str,
    window: str = "1mo",
    *,
    sector_median_pe: Optional[float] = None,
    eps_surprise_pct: Optional[float] = None,
    dcf_score_override: Optional[float] = None,
) -> Dict:
    """
    Backtest the composite-score signal for `ticker` over `window`.

    Parameters
    ----------
    sector_median_pe, eps_surprise_pct, dcf_score_override
        Optional pre-computed values. If omitted, we compute current fundamentals
        once and hold them constant across the window (documented limitation).
    """
    fetch_period, interval, step = _period_for(window)
    ticker = ticker.upper()

    # 1. Historical prices (longer than user-facing window so RSI has warmup)
    hist = yf.Ticker(ticker).history(period=fetch_period, interval=interval, auto_adjust=True)
    if hist.empty:
        return {"window": window, "ticker": ticker, "rows": [], "summary": _empty_summary()}
    closes = hist["Close"].dropna()

    # 2. Current fundamentals (used as a constant proxy for valuation + DCF)
    f_now = compute_fundamentals(ticker)
    val_score_const = f_now.valuation_score if sector_median_pe is None else \
        score_valuation(f_now.trailing_pe, sector_median_pe)[0]
    eps_score_const = score_eps_surprise(
        eps_surprise_pct if eps_surprise_pct is not None else f_now.eps_surprise_pct
    )
    dcf_score_const = dcf_score_override if dcf_score_override is not None else f_now.dcf_score

    rows: List[BacktestRow] = []
    in_window = _trim_to_window(closes, window)
    in_window_index_set = set(in_window.index)

    for i, ts in _iter_checkpoints(closes, step=step):
        if ts not in in_window_index_set:
            continue
        prices_up_to_t = closes.iloc[: i + 1]
        rsi_t = compute_rsi(prices_up_to_t, period=14)
        rsi_score_t = score_rsi(rsi_t)
        momentum_score_t = (rsi_score_t + eps_score_const) / 2.0
        composite_t = (val_score_const + momentum_score_t + dcf_score_const) / 3.0

        if composite_t >= DEFAULT_BUY_THRESHOLD:
            signal = "BUY"
        elif composite_t <= DEFAULT_SELL_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Forward return to the NEXT checkpoint within the file (not calendar-aligned)
        next_idx = min(i + step, len(closes) - 1)
        if next_idx > i:
            fwd = float(closes.iloc[next_idx] / closes.iloc[i] - 1.0)
        else:
            fwd = None

        rows.append(
            BacktestRow(
                date=ts.strftime("%Y-%m-%d"),
                price=round(float(closes.iloc[i]), 4),
                rsi=None if rsi_t is None else round(rsi_t, 2),
                composite=round(composite_t, 2),
                signal=signal,
                fwd_return=None if fwd is None else round(fwd, 6),
            )
        )

    summary = _summarize(rows, in_window)
    return {
        "window": window,
        "ticker": ticker,
        "rows": [r.to_dict() for r in rows],
        "summary": summary,
        "fundamentals_snapshot": {
            "valuation_score": round(val_score_const, 2),
            "eps_surprise_score": round(eps_score_const, 2),
            "dcf_score": round(dcf_score_const, 2),
            "note": "Fundamentals held constant across the window (yfinance free-tier limitation).",
        },
    }


def _empty_summary() -> Dict:
    return {
        "buy_avg_fwd_return": 0.0,
        "hold_avg_fwd_return": 0.0,
        "sell_avg_fwd_return": 0.0,
        "buy_minus_sell": 0.0,
        "accuracy": 0.0,
        "n_buy": 0, "n_hold": 0, "n_sell": 0,
        "cumulative_strategy_return": 0.0,
        "cumulative_buy_and_hold": 0.0,
    }


def _summarize(rows: List[BacktestRow], in_window: pd.Series) -> Dict:
    if not rows:
        return _empty_summary()

    df = pd.DataFrame([r.to_dict() for r in rows])

    def mean_of(mask):
        sub = df.loc[mask, "fwd_return"].dropna()
        return float(sub.mean()) if len(sub) else 0.0

    buy_mask = df["signal"] == "BUY"
    hold_mask = df["signal"] == "HOLD"
    sell_mask = df["signal"] == "SELL"

    buy_avg = mean_of(buy_mask)
    hold_avg = mean_of(hold_mask)
    sell_avg = mean_of(sell_mask)

    # Accuracy: fraction of BUYs where fwd_return > 0 AND fraction of SELLs
    # where fwd_return < 0. Average the two.
    buy_hits = df.loc[buy_mask, "fwd_return"].dropna().gt(0).mean() if buy_mask.any() else 0.0
    sell_hits = df.loc[sell_mask, "fwd_return"].dropna().lt(0).mean() if sell_mask.any() else 0.0
    if buy_mask.any() and sell_mask.any():
        accuracy = float((buy_hits + sell_hits) / 2.0)
    elif buy_mask.any():
        accuracy = float(buy_hits)
    elif sell_mask.any():
        accuracy = float(sell_hits)
    else:
        accuracy = 0.0

    # Cumulative strategy return: compound BUY bars long, SELL bars short, HOLD flat.
    strat_legs = []
    for _, row in df.iterrows():
        if row["fwd_return"] is None or (isinstance(row["fwd_return"], float) and math.isnan(row["fwd_return"])):
            continue
        if row["signal"] == "BUY":
            strat_legs.append(row["fwd_return"])
        elif row["signal"] == "SELL":
            strat_legs.append(-row["fwd_return"])
        else:
            strat_legs.append(0.0)
    strat_cum = float(np.prod([1 + x for x in strat_legs]) - 1) if strat_legs else 0.0

    # Buy-and-hold over the user-facing window
    if len(in_window) >= 2:
        bh = float(in_window.iloc[-1] / in_window.iloc[0] - 1)
    else:
        bh = 0.0

    return {
        "buy_avg_fwd_return": round(buy_avg, 6),
        "hold_avg_fwd_return": round(hold_avg, 6),
        "sell_avg_fwd_return": round(sell_avg, 6),
        "buy_minus_sell": round(buy_avg - sell_avg, 6),
        "accuracy": round(accuracy, 4),
        "n_buy": int(buy_mask.sum()),
        "n_hold": int(hold_mask.sum()),
        "n_sell": int(sell_mask.sum()),
        "cumulative_strategy_return": round(strat_cum, 6),
        "cumulative_buy_and_hold": round(bh, 6),
    }


def run_multi_window_backtest(ticker: str) -> Dict[str, Dict]:
    """Convenience: run 1mo / 1y / 5y in one call."""
    out: Dict[str, Dict] = {}
    for w in ("1mo", "1y", "5y"):
        try:
            out[w] = run_backtest(ticker, window=w)
        except Exception as exc:  # noqa: BLE001
            log.warning("backtest(%s, %s) failed: %s", ticker, w, exc)
            out[w] = {"window": w, "ticker": ticker, "rows": [], "summary": _empty_summary(), "error": str(exc)}
    return out
