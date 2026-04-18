"""
fetcher.py — yfinance data fetching, financial metrics, and signal computation.

Public API
----------
get_stock_info(ticker)        → dict of current price + 10 financial metrics
refresh_signals(tickers, macro_vote) → dict {ticker: {signal, score, votes, details}}
get_weekly_history(ticker, macro_vote) → list of past-5-day signal-vs-outcome dicts

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
    Fetch current price and key financial metrics for *ticker*.

    Strategy (most-reliable first):
      1. fast_info  — lightweight yfinance endpoint; always works on cloud IPs.
                      Provides price, 52-week range, market cap.
      2. .info      — full metadata dict; can be blocked or slow on cloud IPs.
                      Provides financial ratios (P/E, EPS, margins, etc.)
                      and company name / sector.
      3. history()  — fallback for current price and previous close if both
                      fast_info and .info fail.

    All fields default to None; callers render None as "—".
    """
    t = yf.Ticker(ticker)

    result: dict = {
        "company_name": None, "sector": None,
        "current_price": None, "previous_close": None,
        "pe_ratio": None, "eps": None, "market_cap": None,
        "revenue": None, "gross_margin": None, "debt_to_equity": None,
        "roe": None, "pb_ratio": None,
        "week_52_high": None, "week_52_low": None,
        "dividend_yield": None,
    }

    # ── 1. fast_info — reliable on cloud IPs ─────────────────────────────────
    try:
        fi = t.fast_info
        def _fi(attr):
            try:
                v = getattr(fi, attr, None)
                return None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else v
            except Exception:
                return None

        result["current_price"]  = _fi("last_price")
        result["previous_close"] = _fi("previous_close")
        result["market_cap"]     = _fi("market_cap")
        result["week_52_high"]   = _fi("fifty_two_week_high")
        result["week_52_low"]    = _fi("fifty_two_week_low")
    except Exception as exc:
        logger.warning("get_stock_info(%s) fast_info failed: %s", ticker, exc)

    # ── 2. .info — financial ratios + name/sector ─────────────────────────────
    try:
        info = _with_retry(lambda: t.info, retries=2, base_delay=1.0)

        def _get(key):
            val = info.get(key)
            return None if val in (None, "N/A", float("inf"), float("-inf")) else val

        result["company_name"] = _get("longName") or _get("shortName")
        result["sector"]       = _get("sector")

        # Fallback price fields if fast_info missed them
        if result["current_price"] is None:
            result["current_price"] = _get("currentPrice") or _get("regularMarketPrice")
        if result["previous_close"] is None:
            result["previous_close"] = _get("previousClose") or _get("regularMarketPreviousClose")
        if result["market_cap"] is None:
            result["market_cap"] = _get("marketCap")
        if result["week_52_high"] is None:
            result["week_52_high"] = _get("fiftyTwoWeekHigh")
        if result["week_52_low"] is None:
            result["week_52_low"] = _get("fiftyTwoWeekLow")

        result["pe_ratio"]       = _get("trailingPE")
        result["eps"]            = _get("trailingEps")
        result["revenue"]        = _get("totalRevenue")
        result["gross_margin"]   = _get("grossMargins")
        result["debt_to_equity"] = _get("debtToEquity")
        result["roe"]            = _get("returnOnEquity")
        result["pb_ratio"]       = _get("priceToBook")
        result["dividend_yield"] = _get("dividendYield")

    except Exception as exc:
        logger.warning("get_stock_info(%s) .info failed: %s", ticker, exc)

    # ── 3. history fallback — if price still missing ──────────────────────────
    if result["current_price"] is None:
        try:
            hist = _with_retry(
                lambda: t.history(period="5d", interval="1d"),
                retries=2, base_delay=1.0,
            )
            if hist is not None and not hist.empty:
                result["current_price"] = float(hist["Close"].iloc[-1])
                if result["previous_close"] is None and len(hist) >= 2:
                    result["previous_close"] = float(hist["Close"].iloc[-2])
        except Exception as exc:
            logger.warning("get_stock_info(%s) history fallback failed: %s", ticker, exc)

    return result


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

def compute_signal_single(ticker: str, macro_vote: int = 0) -> dict:
    """
    Fetch live intraday + daily data for a single ticker and compute its signal.
    Used for on-demand signal generation when a stock is selected in the UI.
    """
    t = yf.Ticker(ticker)

    # Today's 1-minute bars
    try:
        raw = _with_retry(lambda: t.history(period="1d", interval="1m"), retries=3, base_delay=1.0)
        intraday = _to_et(raw) if (raw is not None and not raw.empty) else None
    except Exception as exc:
        logger.warning("compute_signal_single(%s): intraday fetch failed: %s", ticker, exc)
        intraday = None

    bars_10 = _extract_first_10_min(intraday) if intraday is not None else None

    # 30-day daily bars for prev_close and avg_volume
    try:
        daily = _with_retry(lambda: t.history(period="30d", interval="1d"), retries=3, base_delay=1.0)
        if daily is not None and not daily.empty and len(daily) >= 2:
            prev_close = float(daily["Close"].iloc[-2])
            avg_volume = float(daily["Volume"].mean())
        else:
            prev_close = None
            avg_volume = None
    except Exception as exc:
        logger.warning("compute_signal_single(%s): daily fetch failed: %s", ticker, exc)
        prev_close = None
        avg_volume = None

    return _compute_signal(bars_10, prev_close, avg_volume, macro_vote)


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


# ── 1-Week lookback ───────────────────────────────────────────────────────────

def get_weekly_history(ticker: str, macro_vote: int = 0) -> list[dict]:
    """
    Compute retrospective signals for the last 5 completed trading days and
    compare each signal against that day's actual open-to-close return.

    For each past day this returns:
      date              — ISO date string (YYYY-MM-DD)
      signal            — BUY / HOLD / SELL (computed from first 10 min bars)
      score             — raw score (-5 to +5)
      open              — first market-open price that day
      close             — last market price that day
      day_return_pct    — (close / open - 1) × 100
      signal_return_pct — return you would have captured by following the signal
                          (positive = made money, negative = lost money)
      profitable        — True / False / None (None for HOLD)
    """
    try:
        t = yf.Ticker(ticker)

        # 1-minute bars for the past 7 calendar days (covers ~5 trading days)
        intraday_raw = _with_retry(
            lambda: t.history(period="7d", interval="1m"),
            retries=2, base_delay=1.5,
        )
        if intraday_raw is None or intraday_raw.empty:
            logger.info("get_weekly_history(%s): no intraday data", ticker)
            return []
        intraday = _to_et(intraday_raw)

        # 30-day daily bars for previous-close reference and average volume
        daily_raw = _with_retry(
            lambda: t.history(period="30d", interval="1d"),
            retries=2, base_delay=1.5,
        )
        if daily_raw is None or daily_raw.empty:
            logger.info("get_weekly_history(%s): no daily data", ticker)
            return []

        avg_volume = float(daily_raw["Volume"].mean())

        # Normalise daily index to plain date objects for comparison
        daily_dates = [pd.Timestamp(d).date() for d in daily_raw.index]
        daily_closes = list(daily_raw["Close"])

        # All unique trading days present in the 1-min data, excluding today
        today = datetime.now(ET).date()
        all_days = sorted(set(intraday.index.date))
        past_days = [d for d in all_days if d < today][-5:]

        results: list[dict] = []
        for day in past_days:
            day_bars = intraday[intraday.index.date == day]
            if day_bars.empty:
                continue

            # Previous close = most recent daily close strictly before this day
            before = [(i, c) for i, (d, c) in enumerate(zip(daily_dates, daily_closes)) if d < day]
            prev_close = float(before[-1][1]) if before else None

            # First-10-minute bars for signal computation
            bars_10 = _extract_first_10_min(day_bars)

            sig = _compute_signal(bars_10, prev_close, avg_volume, macro_vote)

            # Day open / close (market-hours bars only)
            market_bars = day_bars[day_bars.index.time >= MARKET_OPEN]
            if market_bars.empty:
                continue
            day_open  = float(market_bars["Open"].iloc[0])
            day_close = float(market_bars["Close"].iloc[-1])
            day_return_pct = round(
                (day_close - day_open) / day_open * 100 if day_open > 0 else 0.0, 2
            )

            signal = sig["signal"]
            if signal == BUY:
                signal_return_pct = day_return_pct
                profitable: Optional[bool] = day_return_pct > 0
            elif signal == SELL:
                signal_return_pct = -day_return_pct
                profitable = day_return_pct < 0
            else:
                signal_return_pct = 0.0
                profitable = None

            results.append({
                "date": day.isoformat(),
                "signal": signal,
                "score": sig["score"],
                "open": round(day_open, 2),
                "close": round(day_close, 2),
                "day_return_pct": day_return_pct,
                "signal_return_pct": round(signal_return_pct, 2),
                "profitable": profitable,
            })

        logger.info("get_weekly_history(%s): %d days returned", ticker, len(results))
        return results

    except Exception as exc:
        logger.warning("get_weekly_history(%s) failed: %s", ticker, exc)
        return []
