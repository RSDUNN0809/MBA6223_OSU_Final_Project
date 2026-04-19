"""
alpha_ranker.py
---------------
Ranks an S&P 500 universe by 12-month trailing alpha relative to the index
(default benchmark: ^GSPC), then surfaces the top N outperformers and bottom N
underperformers. This is the pre-screen that selects the 20 stocks that the
composite scoring pipeline drills into.

Alpha definition (simple, academically defensible):

    alpha_i = total_return_i(12m)  -  total_return_benchmark(12m)

We use adjusted close (dividends + splits reinvested) so the comparison is
apples-to-apples with the benchmark total return.

For efficiency, `yf.download(...)` pulls every ticker in one batched HTTP call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


@dataclass
class AlphaRow:
    ticker: str
    total_return_12m: float
    benchmark_return_12m: float
    alpha_12m: float

    def to_dict(self) -> Dict:
        return asdict(self)


def _total_return_from_adj_close(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    if len(s) < 2:
        return None
    first = float(s.iloc[0])
    last = float(s.iloc[-1])
    if first <= 0:
        return None
    return (last / first) - 1.0


def compute_alpha_table(
    tickers: List[str],
    benchmark: str = "^GSPC",
    lookback: str = "1y",
) -> List[AlphaRow]:
    """
    Pull 12-month adjusted-close data for all tickers in one batch, compute the
    trailing total return for each, and subtract the benchmark return to get
    alpha. Returns one AlphaRow per successfully-priced ticker.
    """
    symbols = list(dict.fromkeys(tickers + [benchmark]))  # dedupe, preserve order
    log.info("downloading %d symbols for alpha ranking", len(symbols))
    data = yf.download(
        tickers=symbols,
        period=lookback,
        interval="1d",
        auto_adjust=True,      # so Close == dividend-adjusted close
        group_by="ticker",
        threads=True,
        progress=False,
    )

    # yf.download returns a multi-indexed column df when multiple tickers given.
    def closes_for(sym: str) -> Optional[pd.Series]:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if sym in data.columns.levels[0]:
                    return data[sym]["Close"]
                return None
            # Single-ticker response
            return data["Close"] if "Close" in data.columns else None
        except Exception as exc:  # noqa: BLE001
            log.debug("closes_for(%s) failed: %s", sym, exc)
            return None

    bench_series = closes_for(benchmark)
    if bench_series is None:
        bench_series = pd.Series(dtype=float)
    bench_ret = _total_return_from_adj_close(bench_series)
    if bench_ret is None:
        raise RuntimeError(f"Could not compute return for benchmark {benchmark}")

    rows: List[AlphaRow] = []
    for t in tickers:
        s = closes_for(t)
        if s is None:
            continue
        r = _total_return_from_adj_close(s)
        if r is None:
            continue
        rows.append(
            AlphaRow(
                ticker=t,
                total_return_12m=round(r, 6),
                benchmark_return_12m=round(bench_ret, 6),
                alpha_12m=round(r - bench_ret, 6),
            )
        )
    return rows


def rank_trailing_alpha(
    tickers: List[str],
    benchmark: str = "^GSPC",
    top_n: int = 10,
    bottom_n: int = 10,
) -> Dict[str, List[AlphaRow]]:
    """
    Returns:
        {
            "top":    [AlphaRow, ...]   # top_n outperformers, sorted desc by alpha
            "bottom": [AlphaRow, ...]   # bottom_n underperformers, sorted asc
            "all":    [AlphaRow, ...]   # full ranked list, desc by alpha
        }
    """
    table = compute_alpha_table(tickers, benchmark=benchmark)
    sorted_desc = sorted(table, key=lambda r: r.alpha_12m, reverse=True)
    top = sorted_desc[:top_n]
    bottom = sorted(sorted_desc[-bottom_n:], key=lambda r: r.alpha_12m)
    return {"top": top, "bottom": bottom, "all": sorted_desc}
