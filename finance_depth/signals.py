"""
signals.py
----------
Top-level facade used by the Flask app. One call gets you everything the
dashboard needs for the three-pillar composite pipeline:

    * 12-month trailing alpha ranking (top 10 / bottom 10 / full table)
    * Per-stock Fundamentals + CompositeResult for the 20 highlighted tickers
    * Current Google Trends threshold adjustment

Designed to hook into the existing _daily_refresh() in app.py so the expensive
yfinance batch pulls only happen once per trading day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

from .fundamentals import (
    Fundamentals,
    build_sector_median_pe_table,
    compute_fundamentals,
    fetch_risk_free_rate,
)
from .composite import (
    CompositeResult,
    DEFAULT_BUY_THRESHOLD,
    DEFAULT_SELL_THRESHOLD,
    compute_composite_score,
)
from .alpha_ranker import AlphaRow, rank_trailing_alpha
from .sentiment_modifier import TrendSentiment, compute_trend_sentiment

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    generated_at: str
    benchmark: str
    trend: Dict
    alpha_top: List[Dict]
    alpha_bottom: List[Dict]
    scored: Dict[str, Dict]   # ticker -> {fundamentals, composite}
    universe_size: int

    def to_dict(self) -> Dict:
        return asdict(self)


def generate_signal(
    ticker: str,
    *,
    sector_median_pe_lookup: Optional[Dict[str, float]] = None,
    trend: Optional[TrendSentiment] = None,
) -> Dict:
    """
    Single-ticker convenience wrapper — useful for the /api/stock/<ticker>
    drill-down route. Returns a JSON-ready dict.
    """
    f = compute_fundamentals(ticker, sector_median_pe_lookup=sector_median_pe_lookup)
    trend = trend or compute_trend_sentiment()
    c = compute_composite_score(
        f,
        trend_adjustment=trend.threshold_shift,
    )
    return {
        "fundamentals": f.to_dict(),
        "composite": c.to_dict(),
        "trend": trend.to_dict(),
    }


def run_full_pipeline(
    universe_tickers: List[str],
    *,
    benchmark: str = "^GSPC",
    top_n: int = 10,
    bottom_n: int = 10,
    trend: Optional[TrendSentiment] = None,
    ticker_sector_map: Optional[Dict[str, str]] = None,
) -> PipelineResult:
    """
    Full daily pipeline. Call this from _daily_refresh().

    Steps:
        1. Rank the whole universe by 12-month trailing alpha vs benchmark.
        2. Pick top 10 + bottom 10 (the 20 "highlighted" tickers).
        3. Build sector-median P/E lookup from the universe (one-time).
        4. Compute Fundamentals + Composite for the 20 tickers.
        5. Attach the current Google Trends threshold shift to every composite.

    If `ticker_sector_map` is provided (built from src.universe.get_sp500()'s
    `ticker` + `sector` columns), sector lookups skip the extra yfinance call.
    """
    log.info("[pipeline] ranking %d tickers vs %s", len(universe_tickers), benchmark)
    ranked = rank_trailing_alpha(universe_tickers, benchmark=benchmark, top_n=top_n, bottom_n=bottom_n)
    top_tickers = [r.ticker for r in ranked["top"]]
    bottom_tickers = [r.ticker for r in ranked["bottom"]]
    highlighted = list(dict.fromkeys(top_tickers + bottom_tickers))

    log.info("[pipeline] building sector-median P/E table")
    sector_pe_lookup = build_sector_median_pe_table(
        universe_tickers,
        ticker_sector_map=ticker_sector_map,
    )

    if trend is None:
        log.info("[pipeline] pulling Google Trends sentiment")
        trend = compute_trend_sentiment()

    scored: Dict[str, Dict] = {}
    for t in highlighted:
        try:
            f = compute_fundamentals(
                t,
                sector_median_pe_lookup=sector_pe_lookup,
                risk_free_rate=fetch_risk_free_rate(),
            )
            c = compute_composite_score(f, trend_adjustment=trend.threshold_shift)
            scored[t] = {"fundamentals": f.to_dict(), "composite": c.to_dict()}
        except Exception as exc:  # noqa: BLE001
            log.warning("[pipeline] scoring failed for %s: %s", t, exc)
            scored[t] = {"error": str(exc)}

    return PipelineResult(
        generated_at=datetime.utcnow().isoformat() + "Z",
        benchmark=benchmark,
        trend=trend.to_dict(),
        alpha_top=[r.to_dict() for r in ranked["top"]],
        alpha_bottom=[r.to_dict() for r in ranked["bottom"]],
        scored=scored,
        universe_size=len(universe_tickers),
    )
