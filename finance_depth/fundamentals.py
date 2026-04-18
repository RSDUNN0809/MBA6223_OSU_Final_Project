"""
fundamentals.py
---------------
Computes the per-stock fundamental inputs that feed the three-pillar composite:

    Valuation pillar  : trailing P/E relative to GICS sector median
    Momentum pillar   : 14-day RSI on daily closes + trailing EPS surprise %
    Intrinsic pillar  : 5-year DCF with CAPM-derived cost of equity

Everything is sourced from yfinance free endpoints. Missing inputs degrade
gracefully: each metric returns a neutral score (50) rather than raising, so
partial data never crashes the pipeline.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# -----------------------------
# Academic / modeling constants
# -----------------------------
EQUITY_RISK_PREMIUM = 0.055        # Damodaran implied ERP, long-run average
TERMINAL_GROWTH_RATE = 0.025       # long-run US GDP growth proxy
DCF_HORIZON_YEARS = 5
DEFAULT_RISK_FREE_RATE = 0.043     # overridden at runtime by ^TNX fetch
MAX_FCF_GROWTH = 0.25              # cap runaway growth assumptions
MIN_KE = 0.06                      # floor on cost of equity to avoid div-by-zero
DEFAULT_BETA = 1.0                 # if beta missing
DEFAULT_SECTOR_PE = 20.0           # if the whole sector has no data


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Fundamentals:
    ticker: str
    sector: Optional[str] = None
    current_price: Optional[float] = None

    # Valuation inputs
    trailing_pe: Optional[float] = None
    sector_median_pe: Optional[float] = None
    valuation_ratio: Optional[float] = None   # stock_PE / sector_median_PE
    valuation_score: float = 50.0

    # Momentum inputs
    rsi_14: Optional[float] = None
    rsi_score: float = 50.0
    eps_surprise_pct: Optional[float] = None
    eps_score: float = 50.0
    momentum_score: float = 50.0

    # Intrinsic value / DCF inputs
    beta: Optional[float] = None
    risk_free_rate: Optional[float] = None
    cost_of_equity: Optional[float] = None
    free_cash_flow: Optional[float] = None
    revenue_growth: Optional[float] = None
    total_debt: Optional[float] = None
    total_cash: Optional[float] = None
    shares_outstanding: Optional[float] = None
    intrinsic_value_per_share: Optional[float] = None
    margin_of_safety: Optional[float] = None
    dcf_score: float = 50.0

    # Diagnostic metadata
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# -----------------------------
# Small numeric helpers
# -----------------------------
def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _is_finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _safe_float(x) -> Optional[float]:
    if not _is_finite(x):
        return None
    return float(x)


# -----------------------------
# Risk-free rate (^TNX = 10-year Treasury yield in %)
# -----------------------------
@lru_cache(maxsize=1)
def fetch_risk_free_rate() -> float:
    """^TNX is quoted in percent times 10 (e.g. 43 = 4.3%)."""
    try:
        hist = yf.Ticker("^TNX").history(period="5d", auto_adjust=False)
        if hist.empty:
            return DEFAULT_RISK_FREE_RATE
        last = float(hist["Close"].dropna().iloc[-1])
        # ^TNX returns e.g. 4.30 meaning 4.30%. Occasionally it's scaled /10.
        if last > 20:
            last = last / 10.0
        return last / 100.0
    except Exception as exc:  # noqa: BLE001
        log.warning("risk-free fetch failed: %s", exc)
        return DEFAULT_RISK_FREE_RATE


# -----------------------------
# Valuation pillar: P/E vs sector median
# -----------------------------
def compute_sector_median_pe(peer_pes: List[float]) -> float:
    values = [p for p in peer_pes if _is_finite(p) and 0 < p < 500]
    if not values:
        return DEFAULT_SECTOR_PE
    return float(np.median(values))


def score_valuation(stock_pe: Optional[float], sector_median_pe: float) -> (float, Optional[float]):
    """Map stock_PE / sector_median_PE to a 0-100 score (higher = cheaper)."""
    if not _is_finite(stock_pe) or stock_pe <= 0 or sector_median_pe <= 0:
        return 50.0, None
    ratio = stock_pe / sector_median_pe
    # ratio = 0.5 (half sector) -> score 75 ; ratio = 1.0 -> 50 ; ratio = 2.0 -> 0
    raw = 100.0 * (1.0 - ratio / 2.0)
    return _clip(raw, 0.0, 100.0), ratio


# -----------------------------
# Momentum pillar: RSI(14) + EPS surprise
# -----------------------------
def compute_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """Classic Wilder RSI on a series of daily closes."""
    if prices is None or len(prices) < period + 1:
        return None
    delta = prices.diff().dropna()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = (-delta.clip(upper=0)).rolling(period).mean()
    if losses.empty or gains.empty:
        return None
    last_gain = gains.iloc[-1]
    last_loss = losses.iloc[-1]
    if not _is_finite(last_gain) or not _is_finite(last_loss):
        return None
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def score_rsi(rsi: Optional[float]) -> float:
    """
    RSI 50 -> neutral score 50.
    Oversold (RSI low) -> bullish -> higher score.
    Overbought (RSI high) -> bearish -> lower score.
    """
    if not _is_finite(rsi):
        return 50.0
    # Linear map: RSI 30 -> 70 (bullish mean-reversion), RSI 70 -> 30 (bearish).
    score = 100.0 - rsi  # RSI 30 -> 70, RSI 70 -> 30
    return _clip(score, 0.0, 100.0)


def score_eps_surprise(surprise_pct: Optional[float]) -> float:
    """
    +20% earnings beat -> 80 ; -20% miss -> 20. Clipped 0..100.
    Empirically, post-earnings announcement drift responds most strongly in
    the first +/- 15% range, so we use a slope of 1.5.
    """
    if not _is_finite(surprise_pct):
        return 50.0
    return _clip(50.0 + surprise_pct * 1.5, 0.0, 100.0)


def compute_eps_surprise_pct(tkr: yf.Ticker) -> Optional[float]:
    """Try multiple yfinance surfaces — earnings data moves between versions."""
    # earnings_history (newer yfinance)
    try:
        eh = getattr(tkr, "earnings_history", None)
        if isinstance(eh, pd.DataFrame) and not eh.empty:
            for col in ("surprisePercent", "Surprise(%)", "Surprise %", "surprise_percent"):
                if col in eh.columns:
                    val = pd.to_numeric(eh[col], errors="coerce").dropna()
                    if not val.empty:
                        last = float(val.iloc[-1])
                        return last * 100 if abs(last) < 1 else last
            if {"epsEstimate", "epsActual"}.issubset(eh.columns):
                est = pd.to_numeric(eh["epsEstimate"], errors="coerce")
                act = pd.to_numeric(eh["epsActual"], errors="coerce")
                diff = (act - est) / est.abs()
                diff = diff.dropna()
                if not diff.empty:
                    return float(diff.iloc[-1] * 100)
    except Exception as exc:  # noqa: BLE001
        log.debug("earnings_history path failed for %s: %s", tkr.ticker, exc)

    # get_earnings_dates (another newer path)
    try:
        ed = tkr.get_earnings_dates(limit=4)
        if isinstance(ed, pd.DataFrame) and not ed.empty:
            if "Surprise(%)" in ed.columns:
                val = pd.to_numeric(ed["Surprise(%)"], errors="coerce").dropna()
                if not val.empty:
                    return float(val.iloc[0])
            if {"EPS Estimate", "Reported EPS"}.issubset(ed.columns):
                est = pd.to_numeric(ed["EPS Estimate"], errors="coerce")
                act = pd.to_numeric(ed["Reported EPS"], errors="coerce")
                diff = ((act - est) / est.abs() * 100).dropna()
                if not diff.empty:
                    return float(diff.iloc[0])
    except Exception as exc:  # noqa: BLE001
        log.debug("get_earnings_dates path failed for %s: %s", tkr.ticker, exc)

    return None


# -----------------------------
# Intrinsic value pillar: DCF + CAPM
# -----------------------------
def capm_cost_of_equity(beta: float, risk_free_rate: float, erp: float = EQUITY_RISK_PREMIUM) -> float:
    beta = beta if _is_finite(beta) else DEFAULT_BETA
    return max(MIN_KE, risk_free_rate + beta * erp)


def discounted_cash_flow(
    fcf: float,
    growth_rate: float,
    ke: float,
    horizon: int = DCF_HORIZON_YEARS,
    terminal_g: float = TERMINAL_GROWTH_RATE,
) -> float:
    """
    5-year FCF projection discounted at Ke, plus Gordon-growth terminal value.
    Returns total enterprise PV of cash flows (not per-share, not equity yet).
    """
    g = _clip(growth_rate if _is_finite(growth_rate) else 0.05, -0.10, MAX_FCF_GROWTH)

    pv_sum = 0.0
    last_fcf = fcf
    for year in range(1, horizon + 1):
        last_fcf = last_fcf * (1.0 + g)
        pv_sum += last_fcf / ((1.0 + ke) ** year)

    # Terminal value with Gordon growth; guard against ke <= terminal_g
    ke_for_terminal = max(ke, terminal_g + 0.01)
    terminal_fcf = last_fcf * (1.0 + terminal_g)
    terminal_value = terminal_fcf / (ke_for_terminal - terminal_g)
    pv_terminal = terminal_value / ((1.0 + ke) ** horizon)

    return pv_sum + pv_terminal


def score_dcf(intrinsic: Optional[float], price: Optional[float]) -> (float, Optional[float]):
    if not (_is_finite(intrinsic) and _is_finite(price) and price > 0):
        return 50.0, None
    mos = (intrinsic - price) / price
    # +50% upside -> 100, 0% -> 50, -50% downside -> 0
    return _clip(50.0 + mos * 100.0, 0.0, 100.0), mos


# -----------------------------
# Public API: compute everything for one ticker
# -----------------------------
def compute_fundamentals(
    ticker: str,
    *,
    sector_median_pe_lookup: Optional[Dict[str, float]] = None,
    risk_free_rate: Optional[float] = None,
) -> Fundamentals:
    """
    Build a full Fundamentals record for one ticker.

    Arguments
    ---------
    sector_median_pe_lookup
        Pre-computed {sector_name: median_pe} dict so we don't recompute for every
        stock in the same universe. See alpha_ranker / signals for the builder.
    """
    fd = Fundamentals(ticker=ticker.upper())
    rf = risk_free_rate if _is_finite(risk_free_rate) else fetch_risk_free_rate()
    fd.risk_free_rate = rf

    try:
        tkr = yf.Ticker(fd.ticker)
        info = getattr(tkr, "info", {}) or {}
    except Exception as exc:  # noqa: BLE001
        fd.warnings.append(f"yfinance_ticker_failed: {exc}")
        return fd

    # Sector + price
    fd.sector = info.get("sector")
    fd.current_price = _safe_float(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )

    # ----- Valuation -----
    fd.trailing_pe = _safe_float(info.get("trailingPE"))
    if sector_median_pe_lookup and fd.sector in sector_median_pe_lookup:
        fd.sector_median_pe = sector_median_pe_lookup[fd.sector]
    else:
        fd.sector_median_pe = DEFAULT_SECTOR_PE
        fd.warnings.append("sector_median_pe_unknown_defaulted")
    fd.valuation_score, fd.valuation_ratio = score_valuation(
        fd.trailing_pe, fd.sector_median_pe
    )

    # ----- Momentum: RSI -----
    try:
        hist = tkr.history(period="90d", auto_adjust=True)
        fd.rsi_14 = compute_rsi(hist["Close"].dropna(), period=14)
    except Exception as exc:  # noqa: BLE001
        fd.warnings.append(f"rsi_history_failed: {exc}")
    fd.rsi_score = score_rsi(fd.rsi_14)

    # ----- Momentum: EPS surprise -----
    try:
        fd.eps_surprise_pct = compute_eps_surprise_pct(tkr)
    except Exception as exc:  # noqa: BLE001
        fd.warnings.append(f"eps_surprise_failed: {exc}")
    fd.eps_score = score_eps_surprise(fd.eps_surprise_pct)

    fd.momentum_score = (fd.rsi_score + fd.eps_score) / 2.0

    # ----- DCF / CAPM -----
    fd.beta = _safe_float(info.get("beta")) or DEFAULT_BETA
    fd.cost_of_equity = capm_cost_of_equity(fd.beta, rf)
    fd.free_cash_flow = _safe_float(info.get("freeCashflow"))
    fd.revenue_growth = _safe_float(info.get("revenueGrowth"))
    fd.total_debt = _safe_float(info.get("totalDebt")) or 0.0
    fd.total_cash = _safe_float(info.get("totalCash")) or 0.0
    fd.shares_outstanding = _safe_float(info.get("sharesOutstanding"))

    if (
        _is_finite(fd.free_cash_flow)
        and fd.free_cash_flow > 0
        and _is_finite(fd.shares_outstanding)
        and fd.shares_outstanding > 0
    ):
        pv_fcf = discounted_cash_flow(
            fcf=fd.free_cash_flow,
            growth_rate=fd.revenue_growth if _is_finite(fd.revenue_growth) else 0.05,
            ke=fd.cost_of_equity,
        )
        equity_value = pv_fcf + fd.total_cash - fd.total_debt
        fd.intrinsic_value_per_share = equity_value / fd.shares_outstanding
        fd.dcf_score, fd.margin_of_safety = score_dcf(
            fd.intrinsic_value_per_share, fd.current_price
        )
    else:
        fd.warnings.append("dcf_inputs_incomplete")
        fd.dcf_score = 50.0

    return fd


def build_sector_median_pe_table(
    tickers: List[str],
    ticker_sector_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    One pass over the universe. Returns {sector: median(trailingPE)}.

    If `ticker_sector_map` is provided (e.g. from src.universe.get_sp500() which
    already contains the GICS sector column), we skip the extra yfinance call
    for sector and only fetch trailingPE per ticker. That cuts ~500 API calls
    per refresh in half.
    """
    by_sector: Dict[str, List[float]] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
        except Exception as exc:  # noqa: BLE001
            log.debug("pe probe failed for %s: %s", t, exc)
            continue
        sector = (ticker_sector_map or {}).get(t) or info.get("sector")
        pe = info.get("trailingPE")
        if sector and _is_finite(pe) and 0 < pe < 500:
            by_sector.setdefault(sector, []).append(float(pe))
    return {s: compute_sector_median_pe(v) for s, v in by_sector.items()}
