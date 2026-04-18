"""
finance_depth
-------------
Drop-in financial-depth package for the MBA6223 OSU Final Project.

Exposes:
    - fundamentals.compute_fundamentals(ticker)
    - composite.compute_composite_score(fundamentals)
    - alpha_ranker.rank_trailing_alpha(tickers, benchmark="^GSPC")
    - sentiment_modifier.trend_adjusted_thresholds(trend_score)
    - backtest.run_backtest(ticker, window="1mo" | "1y" | "5y")
    - signals.generate_signal(ticker, trend_score=None)
"""

from . import fundamentals
from . import composite
from . import alpha_ranker
from . import sentiment_modifier
from . import backtest
from . import signals

__all__ = [
    "fundamentals",
    "composite",
    "alpha_ranker",
    "sentiment_modifier",
    "backtest",
    "signals",
]
