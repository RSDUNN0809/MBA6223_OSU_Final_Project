"""
sentiment_modifier.py
---------------------
Google Trends weekly search intensity for "finance", "stocks", "bonds",
"stock market" is treated as a BEHAVIORAL SENTIMENT MODIFIER that shifts the
BUY/HOLD/SELL thresholds by ±5 points — exactly as specified in the project
objective.

Semantic contract:

    HIGH retail attention (froth / FOMO):
        thresholds shift UP by +5        -> harder to earn a BUY, easier to SELL.

    LOW retail attention (apathy / fear):
        thresholds shift DOWN by −5      -> easier to BUY (contrarian bullish).

    NEUTRAL:
        thresholds unchanged.

"High" / "low" are defined via a z-score on the latest weekly intensity versus
the trailing 12-month mean/std of the same series.

Default keywords per the spec: ["finance", "stocks", "bonds", "stock market"].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_KEYWORDS = ["finance", "stocks", "bonds", "stock market"]
DEFAULT_Z_HIGH = 1.0   # z-score threshold for "unusually high" attention
DEFAULT_Z_LOW = -1.0   # z-score threshold for "unusually low" attention
THRESHOLD_SHIFT = 5.0  # ± points by spec


@dataclass
class TrendSentiment:
    latest_intensity: float
    mean_12m: float
    std_12m: float
    z_score: float
    classification: str          # "high" | "low" | "neutral"
    threshold_shift: float       # +5 / -5 / 0
    keywords: List[str]
    source: str                  # "pytrends" | "fallback"

    def to_dict(self) -> Dict:
        return asdict(self)


def _classify(z: float) -> (str, float):
    if z >= DEFAULT_Z_HIGH:
        return "high", +THRESHOLD_SHIFT
    if z <= DEFAULT_Z_LOW:
        return "low", -THRESHOLD_SHIFT
    return "neutral", 0.0


def _z_from_series(series: pd.Series) -> (float, float, float, float):
    """Return (latest, mean, std, z)."""
    s = series.dropna().astype(float)
    if len(s) < 10:
        return 0.0, 0.0, 0.0, 0.0
    latest = float(s.iloc[-1])
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    z = 0.0 if std == 0 else (latest - mean) / std
    return latest, mean, std, z


def compute_trend_sentiment(
    keywords: Optional[List[str]] = None,
    timeframe: str = "today 12-m",
) -> TrendSentiment:
    """
    Pulls weekly Google Trends via pytrends, averages the columns across the
    keywords, and returns a TrendSentiment object.

    Graceful fallback: if pytrends is unavailable or 429-throttled, returns a
    neutral TrendSentiment (0-shift) so the pipeline keeps working.
    """
    keywords = keywords or DEFAULT_KEYWORDS

    try:
        from pytrends.request import TrendReq  # type: ignore
        pt = TrendReq(hl="en-US", tz=360)
        pt.build_payload(keywords, timeframe=timeframe)
        df = pt.interest_over_time()
        if df is None or df.empty:
            raise RuntimeError("pytrends returned empty frame")
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        composite = df[keywords].mean(axis=1)
        latest, mean, std, z = _z_from_series(composite)
        cls, shift = _classify(z)
        return TrendSentiment(
            latest_intensity=round(latest, 3),
            mean_12m=round(mean, 3),
            std_12m=round(std, 3),
            z_score=round(z, 3),
            classification=cls,
            threshold_shift=shift,
            keywords=keywords,
            source="pytrends",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("trend sentiment fallback (pytrends unavailable): %s", exc)
        return TrendSentiment(
            latest_intensity=0.0,
            mean_12m=0.0,
            std_12m=0.0,
            z_score=0.0,
            classification="neutral",
            threshold_shift=0.0,
            keywords=keywords,
            source="fallback",
        )


def trend_adjusted_thresholds(
    trend: TrendSentiment,
    base_buy: float = 60.0,
    base_sell: float = 40.0,
) -> Dict[str, float]:
    """
    Apply the ±5 shift to both thresholds.
    """
    shift = trend.threshold_shift
    return {
        "buy": base_buy + shift,
        "sell": base_sell + shift,
        "shift_applied": shift,
        "classification": trend.classification,
    }
