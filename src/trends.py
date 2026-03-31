"""
trends.py — Google Trends data via pytrends.

Public API
----------
get_macro_trends() -> list[dict]
    Returns the 10 fixed finance/macro terms with their relative search
    volume scores (0–100) for the past 7 days.

compute_macro_vote(trends) -> int
    Aggregates trend scores into a single directional vote:
    +1 (bullish), 0 (neutral), -1 (bearish).

Rate limits
-----------
pytrends can be throttled by Google.  We fetch in a single batch (all 10
terms at once) and apply exponential backoff with jitter on failure.
If all retries fail, we return neutral scores (50) so the app stays usable.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Optional

from pytrends.request import TrendReq

logger = logging.getLogger(__name__)

# The 10 fixed macro search terms tracked every day
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

# Terms associated with bullish / bearish market sentiment
_BULLISH_TERMS = {"stock market", "S&P 500", "earnings report"}
_BEARISH_TERMS = {"recession", "unemployment"}

# Neutral fallback score when pytrends is unavailable
_NEUTRAL_SCORE = 50

# Retry settings
_MAX_RETRIES = 4
_BASE_DELAY_S = 2.0


def _fetch_with_retry() -> Optional[dict[str, int]]:
    """
    Fetch relative search volume for all MACRO_TERMS in a single pytrends
    request.  Returns {term: score (0-100)} or None on persistent failure.
    """
    pytrends = TrendReq(hl="en-US", tz=-300)  # tz offset in minutes for ET (UTC-5)

    for attempt in range(_MAX_RETRIES):
        try:
            pytrends.build_payload(
                MACRO_TERMS,
                cat=0,
                timeframe="now 7-d",
                geo="US",
            )
            df = pytrends.interest_over_time()

            if df is None or df.empty:
                raise ValueError("Empty response from pytrends")

            # Drop the 'isPartial' column if present
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            # Average score over the returned time window for each term
            scores: dict[str, int] = {
                term: int(round(df[term].mean())) for term in MACRO_TERMS if term in df.columns
            }
            # Fill any missing terms with neutral score
            for term in MACRO_TERMS:
                scores.setdefault(term, _NEUTRAL_SCORE)

            logger.info("Google Trends fetched successfully.")
            return scores

        except Exception as exc:
            delay = _BASE_DELAY_S * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "pytrends attempt %d/%d failed (%s) — retrying in %.1fs",
                attempt + 1, _MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)

    logger.error("All pytrends retries exhausted — returning neutral fallback scores.")
    return None


def get_macro_trends() -> list[dict]:
    """
    Return a sorted list of dicts for the 10 macro terms.

    Each dict: {"term": str, "score": int (0-100)}
    Sorted by score descending (highest interest first).
    """
    raw_scores = _fetch_with_retry()
    scores = raw_scores if raw_scores else {t: _NEUTRAL_SCORE for t in MACRO_TERMS}

    result = [{"term": term, "score": scores.get(term, _NEUTRAL_SCORE)} for term in MACRO_TERMS]
    result.sort(key=lambda x: x["score"], reverse=True)
    return result


def compute_macro_vote(trends: list[dict]) -> int:
    """
    Aggregate the 10 trend scores into a single directional vote.

    Bullish terms (stock market, S&P 500, earnings report):
        High interest → people are engaged with markets → slight bullish signal.
    Bearish terms (recession, unemployment):
        High interest → fear / negative macro → bearish signal.

    Returns +1 (bullish), -1 (bearish), or 0 (neutral).
    The threshold is 15 percentage points to avoid noise.
    """
    if not trends:
        return 0

    term_scores = {t["term"]: t["score"] for t in trends}

    bullish_avg = sum(term_scores.get(t, _NEUTRAL_SCORE) for t in _BULLISH_TERMS) / len(_BULLISH_TERMS)
    bearish_avg = sum(term_scores.get(t, _NEUTRAL_SCORE) for t in _BEARISH_TERMS) / len(_BEARISH_TERMS)

    diff = (bullish_avg - bearish_avg) / 100.0  # normalise to [-1, +1] range

    if diff > 0.15:
        return 1
    elif diff < -0.15:
        return -1
    return 0
