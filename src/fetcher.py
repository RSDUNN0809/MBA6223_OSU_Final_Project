"""
fetcher.py — yfinance data fetching, financial metrics, and signal computation.

Public API
----------
get_stock_info(ticker)        → dict of current price + 10 financial metrics
refresh_signals(tickers, macro_vote) → dict {ticker: {signal, score, votes, details}}

Signal logic (5 indicators, each ±1 or 0):
  1. Gap         — opening gap vs previous close  (threshold: ±1 %)
  2. Momentum    — first-10-min price return       (threshold: ±0.3 %)
  3. VWAP        — last price vs cumulative VWAP   (threshold: ±0.1 %)
  4. Volume      — first-10-min vol vs expected    (threshold: 1.5×/0.5×)
  5. Macro trend — derived from Google Trends      (passed in as macro_vote)

Score ≥ +2 → BUY  |  Score ≤ −2 → SELL  |  else → HOLD
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from datetime import time as dtime
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")
MARKET_OPEN = dtime(9, 30)

# Signal thresholds
_BUY_THRESHOLD = 2
_SELL_THRESHOLD = -2
_GAP_PCT_THRESHOLD = 1.0
_MOMENTUM_PCT_THRESHOLD = 0.3
_VWAP_PCT_THRESHOLD = 0.1
_VOL_HIGH_RATIO = 1.5
_VOL_LOW_RATIO = 0.5

# Expected fraction of daily volume in the first 10 minutes (open premium)
_EXPECTED_FIRST10_FRAC = (10 / 390) * 1.5

# Batch download sizes
_INTRADAY_BATCH = 100
_DAILY_BATCH = 200

# Parallel fetch workers (for fallback single-ticker path)
_MAX_WORKERS = 10

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"


# ── Retry helper ──────────────────────────────────────────────────────────────

def _with_retry(fn, retries: int = 3, base_delay: float = 1.0):
    """
    Call fn() up to *retries* times with exponential backoff.
    Returns the result or raises the last exception.
    """
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            logger.debug("Attempt %d failed (%s) — retrying in %.1fs", attempt + 1, exc, delay)
            time.sleep(delay)
    raise last_exc


# ── Financial metrics ─────────────────────────────────────────────────────────

def get_stock_info(ticker: str) -> dict:
    """
    Fetch the current price and 10 key financial metrics for *ticker*
    via yfinance.  Returns a flat dict; None values indicate unavailable data.
    """
    try:
        info = _with_retry(lambda: yf.Ticker(ticker).info, retries=3, base_delay=1.0)
    except Exception as exc:
        logger.warning("get_stock_info(%s) failed: %s", ticker, exc)
        return {"error": str(exc)}

    def _get(key):
        val = info.get(key)
        return None if val in (None, "N/A", float("inf"), float("-inf")) else val

    return {
        "company_name": _get("longName") or _get("shortName"),
        "sector": _get("sector"),
        # Current price — try several keys yfinance may populate
        "current_price": (
            _get("currentPrice")
            or _get("regularMarketPrice")
            or _get("ask")
            or _get("bid")
        ),
        "previous_close": _get("previousClose") or _get("regularMarketPreviousClose"),
        # 10 required financial metrics
        "pe_ratio": _get("trailingPE"),
        "eps": _get("trailingEps"),
        "market_cap": _get("marketCap"),
        "revenue": _get("totalRevenue"),
        "gross_margin": _get("grossMargins"),   # decimal, e.g. 0.432
        "debt_to_equity": _get("debtToEquity"),
        "roe": _get("returnOnEquity"),           # decimal, e.g. 1.473
        "pb_ratio": _get("priceToBook"),
        "week_52_high": _get("fiftyTwoWeekHigh"),
        "week_52_low": _get("fiftyTwoWeekLow"),
        "dividend_yield": _get("dividendYield"), # decimal, e.g. 0.0052
    }


# ── Intraday data helpers ──────────────────────────────────────────────────────

def _to_et(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is timezone-aware and in US/Eastern."""
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df


def _extract_first_10_min(bars: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Slice *bars* to the first 10 one-minute candles at or after 9:30 AM ET."""
    if bars is None or bars.empty:
        return None
    market_bars = bars[bars.index.time >= MARKET_OPEN]
    return market_bars.head(10) if not market_bars.empty else None


def _fetch_intraday_single(ticker: str) -> tuple[str, Optional[pd.DataFrame]]:
    """Single-ticker intraday fallback (used when batch download fails)."""
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="1m")
        return ticker, (_to_et(df) if not df.empty else None)
    except Exception as exc:
        logger.debug("Single intraday fetch failed for %s: %s", ticker, exc)
        return ticker, None


def _get_intraday_bars(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Batch-download today's 1-minute bars for all tickers.
    Falls back to parallel single-ticker fetches if the batch fails.
    """
    all_data: dict[str, pd.DataFrame] = {}

    for i in range(0, len(tickers), _INTRADAY_BATCH):
        batch = tickers[i: i + _INTRADAY_BATCH]
        try:
            raw = _with_retry(
                lambda b=batch: yf.download(
                    b, period="1d", interval="1m",
                    group_by="ticker", auto_adjust=True,
                    progress=False, threads=True,
                ),
                retries=3, base_delay=2.0,
            )
            if raw.empty:
                raise ValueError("Empty batch result")

            available = raw.columns.get_level_values(0).unique().tolist()
            for ticker in available:
                try:
                    df = raw[ticker].dropna(how="all")
                    if not df.empty:
                        all_data[ticker] = _to_et(df.copy())
                except Exception:
                    pass

        except Exception as exc:
            logger.warning("Intraday batch %d failed (%s) — falling back to single fetches.", i, exc)
            with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
                futures = {pool.submit(_fetch_intraday_single, t): t for t in batch}
                for future in as_completed(futures):
                    ticker, df = future.result()
                    if df is not None:
                        all_data[ticker] = df

    logger.info("Intraday bars: %d / %d tickers", len(all_data), len(tickers))
    return all_data


def _get_daily_info(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch 30-day daily bars for all tickers.
    Returns {ticker: {"prev_close": float, "avg_volume": float}}.
    """
    result: dict[str, dict] = {}

    for i in range(0, len(tickers), _DAILY_BATCH):
        batch = tickers[i: i + _DAILY_BATCH]
        try:
            raw = _with_retry(
                lambda b=batch: yf.download(
                    b, period="30d", interval="1d",
                    group_by="ticker", auto_adjust=True,
                    progress=False, threads=True,
                ),
                retries=3, base_delay=2.0,
            )
            if raw.empty:
                continue

            available = raw.columns.get_level_values(0).unique().tolist()
            for ticker in available:
                try:
                    df = raw[ticker].dropna(how="all")
                    if len(df) >= 2:
                        result[ticker] = {
                            "prev_close": float(df["Close"].iloc[-2]),
                            "avg_volume": float(df["Volume"].mean()),
                        }
                except Exception:
                    pass

        except Exception as exc:
            logger.warning("Daily batch %d failed: %s", i, exc)

    logger.info("Daily info: %d / %d tickers", len(result), len(tickers))
    return result


# ── VWAP helper ───────────────────────────────────────────────────────────────

def _vwap(bars: pd.DataFrame) -> float:
    """Cumulative VWAP = Σ(typical_price × volume) / Σ(volume)."""
    typical = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    total_vol = bars["Volume"].sum()
    if total_vol == 0:
        return float(bars["Close"].iloc[-1])
    return float((typical * bars["Volume"]).sum() / total_vol)


# ── Signal computation ────────────────────────────────────────────────────────

def _compute_signal(
    bars_10: Optional[pd.DataFrame],
    prev_close: Optional[float],
    avg_daily_volume: Optional[float],
    macro_vote: int = 0,
) -> dict:
    """
    Score a single ticker across 5 indicators and return a BUY/SELL/HOLD signal.

    Returns dict with keys: signal, score, votes, details.
    """
    if bars_10 is None or bars_10.empty:
        return {
            "signal": HOLD, "score": 0,
            "votes": {}, "details": {"error": "no_intraday_data"},
        }

    votes: dict[str, int] = {}
    details: dict = {}

    first_open = float(bars_10["Open"].iloc[0])
    last_close = float(bars_10["Close"].iloc[-1])

    # 1. Gap ──────────────────────────────────────────────────────────────────
    if prev_close and prev_close > 0:
        gap_pct = (first_open - prev_close) / prev_close * 100.0
        details["gap_pct"] = round(gap_pct, 2)
        votes["gap"] = 1 if gap_pct >= _GAP_PCT_THRESHOLD else (-1 if gap_pct <= -_GAP_PCT_THRESHOLD else 0)
    else:
        details["gap_pct"] = None
        votes["gap"] = 0

    # 2. Momentum ─────────────────────────────────────────────────────────────
    momentum_pct = (last_close - first_open) / first_open * 100.0 if first_open > 0 else 0.0
    details["momentum_pct"] = round(momentum_pct, 2)
    votes["momentum"] = 1 if momentum_pct >= _MOMENTUM_PCT_THRESHOLD else (-1 if momentum_pct <= -_MOMENTUM_PCT_THRESHOLD else 0)

    # 3. VWAP ─────────────────────────────────────────────────────────────────
    vwap_val = _vwap(bars_10)
    details["vwap"] = round(vwap_val, 4)
    details["last_price"] = round(last_close, 4)
    price_vs_vwap = (last_close - vwap_val) / vwap_val * 100.0 if vwap_val > 0 else 0.0
    details["price_vs_vwap_pct"] = round(price_vs_vwap, 3)
    votes["vwap"] = 1 if price_vs_vwap >= _VWAP_PCT_THRESHOLD else (-1 if price_vs_vwap <= -_VWAP_PCT_THRESHOLD else 0)

    # 4. Volume ───────────────────────────────────────────────────────────────
    first_10_vol = float(bars_10["Volume"].sum())
    if avg_daily_volume and avg_daily_volume > 0:
        expected = avg_daily_volume * _EXPECTED_FIRST10_FRAC
        vol_ratio = first_10_vol / expected
        details["vol_ratio"] = round(vol_ratio, 2)
        # High volume confirms momentum direction; thin tape → neutral
        votes["volume"] = votes.get("momentum", 0) if vol_ratio >= _VOL_HIGH_RATIO else 0
    else:
        details["vol_ratio"] = None
        votes["volume"] = 0

    # 5. Macro trends (pre-computed Google Trends sentiment vote) ─────────────
    votes["macro_trend"] = int(macro_vote)

    # Aggregate ───────────────────────────────────────────────────────────────
    score = sum(votes.values())
    signal = BUY if score >= _BUY_THRESHOLD else (SELL if score <= _SELL_THRESHOLD else HOLD)

    return {"signal": signal, "score": score, "votes": votes, "details": details}


# ── Batch signal refresh ──────────────────────────────────────────────────────

def refresh_signals(tickers: list[str], macro_vote: int = 0) -> dict[str, dict]:
    """
    Download intraday + daily data for all *tickers* and compute signals.
    Returns {ticker: {signal, score, votes, details}}.

    This is called once per day at 9:40 AM ET and takes several minutes
    for the full S&P 500.  Results are cached by the caller.
    """
    logger.info("Refreshing signals for %d tickers (macro_vote=%d)...", len(tickers), macro_vote)

    intraday = _get_intraday_bars(tickers)
    daily = _get_daily_info(tickers)

    signals: dict[str, dict] = {}
    for ticker in tickers:
        bars = intraday.get(ticker)
        bars_10 = _extract_first_10_min(bars)
        d = daily.get(ticker, {})
        signals[ticker] = _compute_signal(
            bars_10,
            prev_close=d.get("prev_close"),
            avg_daily_volume=d.get("avg_volume"),
            macro_vote=macro_vote,
        )

    buy_ct = sum(1 for v in signals.values() if v["signal"] == BUY)
    sell_ct = sum(1 for v in signals.values() if v["signal"] == SELL)
    hold_ct = sum(1 for v in signals.values() if v["signal"] == HOLD)
    logger.info("Signals complete — BUY: %d  SELL: %d  HOLD: %d", buy_ct, sell_ct, hold_ct)
    return signals
