"""
trends.py — Market-based macro sentiment indicators.

Replaces pytrends (unreliable from cloud hosts due to Google rate-limiting)
with yfinance market proxies. Each macro term maps to a ticker whose recent
behaviour serves as a proxy for how "hot" that topic is in markets right now.

Proxy logic
-----------
  return_5d   : 5-day total return → 50 = flat, +5% → 100, -5% → 0
  return_20d  : 20-day return → 50 = flat, +10% → 100, -10% → 0
  level       : current value as % of its 52-week range (0 = yearly low, 100 = high)
  level_inv   : inverted level (higher raw value → lower score, e.g. recession fear)
  change_5d   : absolute 5-day change in level, scaled to 0-100
  vol_ratio   : today's volume vs 20-day avg volume, scaled to 0-100

Scores are clipped to [0, 100] and rounded to the nearest integer.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Fixed term order (display order in the UI)
MACRO_TERMS: list[str] = [
    "stock market",
    "inflation",
    "Federal Reserve",
    "interest rates",
    "recession",
    "S&P 500",
    "unemployment",
    "GDP",
    "earnings report",
    "oil prices",
]

# (ticker, method) for each term
_PROXIES: dict[str, tuple[str, str]] = {
    "stock market":    ("^GSPC", "return_5d"),
    "inflation":       ("^TNX",  "level"),        # 10-yr yield: high yield = high inflation concern
    "Federal Reserve": ("^IRX",  "level"),        # 3-month T-bill: proxy for fed funds expectation
    "interest rates":  ("^TNX",  "change_5d"),    # recent move in yields
    "recession":       ("^VIX",  "level_inv"),    # VIX high = fear high, inverted so high fear = high score
    "S&P 500":         ("^GSPC", "return_20d"),   # broader 20-day momentum
    "unemployment":    ("^VIX",  "change_5d"),    # rising VIX → growing worry about labour market
    "GDP":             ("SPY",   "return_20d"),   # broad market as growth proxy
    "earnings report": ("SPY",   "vol_ratio"),    # volume spike = heightened earnings activity
    "oil prices":      ("CL=F",  "return_5d"),    # crude oil 5-day move
}

_NEUTRAL = 50
_TICKERS = list(dict.fromkeys(t for t, _ in _PROXIES.values()))
_BULLISH_TERMS = {"stock market", "S&P 500", "earnings report"}
_BEARISH_TERMS = {"recession", "unemployment"}


def _clip(x: float) -> int:
    return int(round(max(0.0, min(100.0, x))))


def _score_return(ret: float, scale: float = 10.0) -> int:
    """ret is a decimal (e.g. 0.03 for +3%). scale maps ±(1/scale) → ±50 pts."""
    return _clip(50.0 + ret * scale * 100.0)


def _score_level(series: pd.Series) -> int:
    """Current value as % of the series' full range."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return _NEUTRAL
    return _clip((series.iloc[-1] - lo) / (hi - lo) * 100.0)


def _score_level_inv(series: pd.Series) -> int:
    return 100 - _score_level(series)


def _score_change(series: pd.Series, window: int = 5, scale: float = 5.0) -> int:
    """Absolute change over `window` bars as a % of level, scaled."""
    if len(series) < window + 1:
        return _NEUTRAL
    recent = series.iloc[-1]
    past   = series.iloc[-(window + 1)]
    if past == 0:
        return _NEUTRAL
    change_pct = (recent - past) / abs(past)
    return _clip(50.0 + change_pct * scale * 100.0)


def _score_vol_ratio(hist: pd.DataFrame, window: int = 20) -> int:
    if "Volume" not in hist.columns or len(hist) < window + 1:
        return _NEUTRAL
    avg_vol = hist["Volume"].iloc[-window - 1:-1].mean()
    if avg_vol == 0:
        return _NEUTRAL
    ratio = hist["Volume"].iloc[-1] / avg_vol   # 1.0 = normal
    return _clip(ratio * 50.0)                   # 2× avg → 100, 0× → 0


def _compute_scores() -> dict[str, int]:
    """
    Download 1-year daily history for all proxy tickers in one batch,
    then compute each term's score. Falls back to _NEUTRAL per term on error.
    """
    scores: dict[str, int] = {}

    try:
        data = yf.download(
            tickers=_TICKERS,
            period="1y",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as exc:
        logger.warning("yfinance batch download failed: %s", exc)
        return {term: _NEUTRAL for term in MACRO_TERMS}

    def get_close(ticker: str) -> Optional[pd.Series]:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                return data[ticker]["Close"].dropna() if ticker in data.columns.get_level_values(0) else None
            return data["Close"].dropna() if len(_TICKERS) == 1 else None
        except Exception:
            return None

    def get_hist(ticker: str) -> Optional[pd.DataFrame]:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                return data[ticker].dropna(how="all") if ticker in data.columns.get_level_values(0) else None
            return data.dropna(how="all") if len(_TICKERS) == 1 else None
        except Exception:
            return None

    for term, (ticker, method) in _PROXIES.items():
        try:
            close = get_close(ticker)
            if close is None or len(close) < 6:
                scores[term] = _NEUTRAL
                continue

            if method == "return_5d":
                ret = (close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0
                scores[term] = _score_return(ret, scale=10.0)

            elif method == "return_20d":
                idx = min(20, len(close) - 1)
                ret = (close.iloc[-1] / close.iloc[-idx - 1] - 1.0)
                scores[term] = _score_return(ret, scale=5.0)

            elif method == "level":
                scores[term] = _score_level(close)

            elif method == "level_inv":
                scores[term] = _score_level_inv(close)

            elif method == "change_5d":
                scores[term] = _score_change(close, window=5)

            elif method == "vol_ratio":
                hist_df = get_hist(ticker)
                scores[term] = _score_vol_ratio(hist_df) if hist_df is not None else _NEUTRAL

            else:
                scores[term] = _NEUTRAL

        except Exception as exc:
            logger.debug("score(%s, %s) failed: %s", term, ticker, exc)
            scores[term] = _NEUTRAL

    return scores


def get_macro_trends() -> list[dict]:
    """
    Return a sorted list of dicts for the 10 macro terms.
    Each dict: {"term": str, "score": int (0-100)}
    Sorted by score descending.
    """
    raw = _compute_scores()
    result = [{"term": t, "score": raw.get(t, _NEUTRAL)} for t in MACRO_TERMS]
    result.sort(key=lambda x: x["score"], reverse=True)
    return result


def compute_macro_vote(trends: list[dict]) -> int:
    """
    Aggregate scores into +1 (bullish), -1 (bearish), 0 (neutral).
    Threshold: 15-point spread between bullish and bearish term averages.
    """
    if not trends:
        return 0
    term_scores = {t["term"]: t["score"] for t in trends}
    bull_avg = sum(term_scores.get(t, _NEUTRAL) for t in _BULLISH_TERMS) / len(_BULLISH_TERMS)
    bear_avg = sum(term_scores.get(t, _NEUTRAL) for t in _BEARISH_TERMS) / len(_BEARISH_TERMS)
    diff = (bull_avg - bear_avg) / 100.0
    if diff > 0.15:
        return 1
    if diff < -0.15:
        return -1
    return 0
