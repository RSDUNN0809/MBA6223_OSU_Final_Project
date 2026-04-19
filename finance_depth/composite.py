"""
composite.py
------------
Rolls the three pillar scores from `fundamentals.Fundamentals` up into a single
0..100 composite. Also produces the BUY / HOLD / SELL signal by comparing the
composite to thresholds that can be shifted by the Google Trends modifier.

Weights are equal by the project spec (1/3 each):

    composite = (valuation_score + momentum_score + dcf_score) / 3
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

from .fundamentals import Fundamentals

# Default thresholds (unmodified). The sentiment modifier shifts BOTH by ±5.
DEFAULT_BUY_THRESHOLD = 60.0
DEFAULT_SELL_THRESHOLD = 40.0


@dataclass
class CompositeResult:
    ticker: str
    valuation_score: float
    momentum_score: float
    dcf_score: float
    composite_score: float
    buy_threshold: float
    sell_threshold: float
    signal: str            # "BUY" | "HOLD" | "SELL"
    trend_adjustment: float  # +5 / 0 / -5 applied to thresholds
    pillar_breakdown: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_composite_score(
    f: Fundamentals,
    *,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
    trend_adjustment: float = 0.0,
) -> CompositeResult:
    """
    Build a CompositeResult.

    `trend_adjustment` shifts BOTH thresholds together:
        +5  -> harder to BUY, easier to SELL (high retail froth)
        -5  -> easier to BUY, harder to SELL (retail fear / disinterest)
    """
    composite = (f.valuation_score + f.momentum_score + f.dcf_score) / 3.0

    adj_buy = buy_threshold + trend_adjustment
    adj_sell = sell_threshold + trend_adjustment

    if composite >= adj_buy:
        signal = "BUY"
    elif composite <= adj_sell:
        signal = "SELL"
    else:
        signal = "HOLD"

    pillar_breakdown = {
        "valuation": round(f.valuation_score, 2),
        "momentum": round(f.momentum_score, 2),
        "intrinsic_value": round(f.dcf_score, 2),
        "rsi_subscore": round(f.rsi_score, 2),
        "eps_surprise_subscore": round(f.eps_score, 2),
    }

    return CompositeResult(
        ticker=f.ticker,
        valuation_score=round(f.valuation_score, 2),
        momentum_score=round(f.momentum_score, 2),
        dcf_score=round(f.dcf_score, 2),
        composite_score=round(composite, 2),
        buy_threshold=adj_buy,
        sell_threshold=adj_sell,
        signal=signal,
        trend_adjustment=trend_adjustment,
        pillar_breakdown=pillar_breakdown,
    )
