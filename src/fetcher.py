"""
fetcher.py — yfinance data fetching, financial metrics, and signal computation.

Public API
----------
get_stock_info(ticker)        → dict of current price + 10 financial metrics + analyst data
refresh_signals(tickers, macro_vote) → dict {ticker: {signal, score, votes, details}}
get_weekly_history(ticker, macro_vote) → list of past-5-day signal-vs-outcome dicts

Signal logic (9 indicators, each ±1 or 0):
  1. Gap         — opening gap vs previous close    (threshold: ±0.1 %)
  2. Momentum    — first-10-min price return         (threshold: ±0.1 %)
  3. VWAP        — last price vs cumulative VWAP     (threshold: ±0.2 %)
  4. Volume      — first-10-min vol vs expected      (threshold: 1.0×)
  5. Macro trend — derived from Google Trends        (passed in as macro_vote)
  6. RSI-14      — daily RSI > 50 = bullish regime   (computed from 3mo bars)
  7. MA-50       — price above 50-day MA = uptrend   (computed from 3mo bars)
  8. Sector ETF  — sector ETF moving same direction  (intraday return)
  9. Analyst     — Wall St. consensus recommendationMean ≤2.5 → +3, ≥3.5 → −3 (3× weight)

Max score: +11 (8 single votes + analyst 3×)  Min score: ≈ −10
Score ≥ +1 → BUY  |  Score ≤ −1 → SELL  |  else → HOLD
VIX gate: BUY suppressed → HOLD when VIX ≥ 25 (high-fear override)
Strategy: BUY = long, SELL/HOLD = flat (no short positions)
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

# Signal thresholds (optimizer v4 — 9-indicator, analyst_w=3)
_BUY_THRESHOLD          = 1
_SELL_THRESHOLD         = -1
_GAP_PCT_THRESHOLD      = 0.5    # higher bar filters noise; analyst weight anchors
_MOMENTUM_PCT_THRESHOLD = 0.3    # same — cleaner confirmation needed
_VWAP_PCT_THRESHOLD     = 0.3
_VOL_HIGH_RATIO         = 2.0    # require stronger volume confirmation
_VOL_LOW_RATIO          = 0.5
_RSI_WINDOW             = 14
_MA_WINDOW              = 50
_VIX_GATE               = 25.0   # keep gate for live risk management despite no optimizer benefit
_ANALYST_WEIGHT         = 3      # analyst vote counts 3× (optimizer: −0.18 alpha vs −0.37 at 0×)

# Expected fraction of daily volume in the first 10 minutes (open premium)
_EXPECTED_FIRST10_FRAC = (10 / 390) * 1.5

# ── Sector ETF map ─────────────────────────────────────────────────────────────
_SECTOR_ETF_MAP: dict[str, str] = {}
def _build_sector_map() -> None:
    _m = _SECTOR_ETF_MAP
    for t in ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","AMD","INTC","QCOM","TXN",
              "ACN","IBM","NOW","AMAT","ADI","PANW","KLAC","LRCX","CDNS","SNPS",
              "MU","MCHP","ROP","FSLR","HPE","KEYS","CTSH","IT","CDW","ANSS"]:
        _m[t] = "XLK"
    for t in ["META","GOOGL","GOOG","NFLX","DIS","CMCSA","T","VZ","CHTR","EA",
              "MTCH","IPG","OMC","PARA","WBD","TTWO","LYV","FOXA","FOX","NWS"]:
        _m[t] = "XLC"
    for t in ["AMZN","TSLA","HD","MCD","NKE","SBUX","TJX","LOW","BKNG","ORLY",
              "AZO","MAR","GM","F","DHI","PHM","ROST","YUM","HLT","EBAY",
              "ABNB","RCL","CCL","NCLH","GPC","KMX","BBY","APTV","BWA","LVS"]:
        _m[t] = "XLY"
    for t in ["WMT","PG","COST","KO","PEP","PM","MO","MDLZ","CL","KMB",
              "KR","HSY","SYY","EL","GIS","CHD","MKC","HRL","CAG","MNST","STZ"]:
        _m[t] = "XLP"
    for t in ["LLY","JNJ","UNH","ABBV","MRK","TMO","ABT","DHR","AMGN","PFE",
              "CVS","CI","ELV","HUM","ISRG","BSX","ZTS","BDX","SYK","REGN",
              "VRTX","MDT","IQV","MCK","CAH","DGX","LH","BAX","A","GEHC"]:
        _m[t] = "XLV"
    for t in ["BRK-B","JPM","V","MA","BAC","WFC","GS","MS","AXP","SPGI",
              "BLK","MMC","CB","PGR","TRV","AFL","MET","PRU","ALL","AIG",
              "ICE","CME","MSCI","FIS","FISV","COF","DFS","ALLY","SYF","CINF"]:
        _m[t] = "XLF"
    for t in ["GE","CAT","RTX","UNP","HON","DE","LMT","BA","UPS","ETN",
              "GD","PH","NSC","ITW","EMR","CTAS","CMI","ROK","PCAR","IR",
              "FDX","XYL","CARR","OTIS","FAST","J","MAS","SWK","PNR","AME"]:
        _m[t] = "XLI"
    for t in ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","HES","HAL",
              "DVN","OXY","CTRA","KMI","WMB","OKE","TRGP","BKR","MRO","APA"]:
        _m[t] = "XLE"
    for t in ["LIN","APD","SHW","ECL","FCX","NEM","VMC","MLM","PPG","ALB",
              "DD","DOW","CE","IFF","MOS","PKG","IP","CF","RPM","FMC"]:
        _m[t] = "XLB"
    for t in ["PLD","AMT","CCI","EQIX","PSA","DLR","O","SPG","WELL","AVB",
              "EQR","ARE","VTR","BXP","KIM","REG","ESS","MAA","UDR","CPT"]:
        _m[t] = "XLRE"
    for t in ["NEE","DUK","SO","D","AEP","EXC","SRE","XEL","ES","WEC",
              "ED","ETR","PPL","FE","EIX","AWK","DTE","CMS","CNP","ATO"]:
        _m[t] = "XLU"
_build_sector_map()

def _sector_etf(ticker: str) -> str:
    return _SECTOR_ETF_MAP.get(ticker, "SPY")

def _compute_rsi(close: pd.Series, window: int = _RSI_WINDOW) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

# ── Analyst data helper ────────────────────────────────────────────────────────

def _fetch_analyst_data(ticker: str) -> dict:
    """
    Fetch analyst consensus fields from .info for a single ticker.
    Returns a dict with recommendation_mean and target prices (all may be None).
    """
    try:
        info = _with_retry(lambda: yf.Ticker(ticker).info, retries=2, base_delay=1.0)
        def _g(key):
            val = info.get(key)
            return None if val in (None, "N/A", float("inf"), float("-inf")) else val
        return {
            "recommendation_mean": _g("recommendationMean"),
            "target_mean_price":   _g("targetMeanPrice"),
            "target_high_price":   _g("targetHighPrice"),
            "target_low_price":    _g("targetLowPrice"),
            "analyst_count":       _g("numberOfAnalystOpinions"),
        }
    except Exception as exc:
        logger.debug("_fetch_analyst_data(%s) failed: %s", ticker, exc)
        return {}


def _fetch_analyst_batch(tickers: list[str]) -> dict[str, dict]:
    """Parallel analyst data fetch for all tickers."""
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_analyst_data, t): t for t in tickers}
        for future in as_completed(futures):
            t = futures[future]
            try:
                results[t] = future.result()
            except Exception:
                results[t] = {}
    return results


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
        # Analyst / forward estimates
        "forward_pe": None, "forward_eps": None,
        "target_mean_price": None, "target_high_price": None, "target_low_price": None,
        "analyst_count": None, "recommendation_mean": None,
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

        # Analyst / forward estimates
        result["forward_pe"]          = _get("forwardPE")
        result["forward_eps"]         = _get("forwardEps")
        result["target_mean_price"]   = _get("targetMeanPrice")
        result["target_high_price"]   = _get("targetHighPrice")
        result["target_low_price"]    = _get("targetLowPrice")
        result["analyst_count"]       = _get("numberOfAnalystOpinions")
        result["recommendation_mean"] = _get("recommendationMean")

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
    Fetch 3-month daily bars for all tickers.
    Returns {ticker: {prev_close, avg_volume, rsi, above_ma50}}.
    Extended to 3mo (≈63 bars) to support MA50 and RSI computation.
    """
    result: dict[str, dict] = {}

    for i in range(0, len(tickers), _DAILY_BATCH):
        batch = tickers[i: i + _DAILY_BATCH]
        try:
            raw = _with_retry(
                lambda b=batch: yf.download(
                    b, period="3mo", interval="1d",
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
                    if len(df) < 2:
                        continue
                    close = df["Close"]
                    rsi_s = _compute_rsi(close)
                    rsi_v = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else None
                    ma50_v = float(close.rolling(_MA_WINDOW).mean().iloc[-1]) if len(close) >= _MA_WINDOW else None
                    result[ticker] = {
                        "prev_close":  float(close.iloc[-2]),
                        "avg_volume":  float(df["Volume"].mean()),
                        "rsi":         rsi_v,
                        "above_ma50":  (float(close.iloc[-1]) > ma50_v) if ma50_v is not None else None,
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


# ── Market context (VIX + sector ETFs) ───────────────────────────────────────

_ALL_SECTOR_ETFS = ["XLK","XLC","XLY","XLP","XLV","XLF","XLI","XLE","XLB","XLRE","XLU","SPY"]

def _fetch_market_context() -> dict:
    """
    Fetch today's intraday data for ^VIX and all sector ETFs in a single batch.
    Returns {"vix": float|None, "etf_returns": {symbol: float|None}}.
    """
    ctx: dict = {"vix": None, "etf_returns": {e: None for e in _ALL_SECTOR_ETFS}}
    symbols = ["^VIX"] + _ALL_SECTOR_ETFS
    try:
        raw = _with_retry(
            lambda: yf.download(
                symbols, period="2d", interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True,
            ),
            retries=2, base_delay=1.5,
        )
        if raw.empty:
            return ctx

        # VIX current level
        try:
            vix_df = raw["^VIX"]["Close"].dropna()
            if not vix_df.empty:
                ctx["vix"] = float(vix_df.iloc[-1])
        except Exception:
            pass

        # Sector ETF today-vs-yesterday return
        for etf in _ALL_SECTOR_ETFS:
            try:
                etf_close = raw[etf]["Close"].dropna()
                if len(etf_close) >= 2:
                    ctx["etf_returns"][etf] = float(etf_close.iloc[-1] / etf_close.iloc[-2] - 1)
            except Exception:
                pass

    except Exception as exc:
        logger.warning("_fetch_market_context failed: %s", exc)
    return ctx


# ── Signal computation ────────────────────────────────────────────────────────

def _compute_signal(
    bars_10: Optional[pd.DataFrame],
    prev_close: Optional[float],
    avg_daily_volume: Optional[float],
    macro_vote: int = 0,
    rsi: Optional[float] = None,
    above_ma50: Optional[bool] = None,
    sector_ret: Optional[float] = None,
    vix_level: Optional[float] = None,
    analyst_rec: Optional[float] = None,
) -> dict:
    """
    Score a single ticker across up to 9 indicators and return BUY/SELL/HOLD.
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
        votes["volume"] = votes.get("momentum", 0) if vol_ratio >= _VOL_HIGH_RATIO else 0
    else:
        details["vol_ratio"] = None
        votes["volume"] = 0

    # 5. Macro trends ─────────────────────────────────────────────────────────
    votes["macro_trend"] = int(macro_vote)

    # 6. RSI regime: > 50 = bullish momentum, < 50 = bearish ─────────────────
    if rsi is not None and np.isfinite(rsi):
        details["rsi"] = round(rsi, 1)
        votes["rsi"] = 1 if rsi > 50 else -1

    # 7. 50-day MA trend filter ───────────────────────────────────────────────
    if above_ma50 is not None:
        details["above_ma50"] = above_ma50
        votes["ma50"] = 1 if above_ma50 else -1

    # 8. Sector ETF: broad sector moving same direction ───────────────────────
    if sector_ret is not None and np.isfinite(sector_ret):
        details["sector_ret"] = round(sector_ret * 100, 2)
        votes["sector"] = 1 if sector_ret > 0.0 else (-1 if sector_ret < 0.0 else 0)

    # 9. Analyst consensus: Wall St. recommendationMean (1=Strong Buy, 5=Sell) ─
    # Weighted _ANALYST_WEIGHT× — optimizer found 3× reduces alpha gap most.
    if analyst_rec is not None and np.isfinite(analyst_rec):
        details["recommendation_mean"] = round(analyst_rec, 2)
        raw_vote = 1 if analyst_rec <= 2.5 else (-1 if analyst_rec >= 3.5 else 0)
        votes["analyst"] = raw_vote * _ANALYST_WEIGHT

    # Aggregate ───────────────────────────────────────────────────────────────
    score = sum(votes.values())
    signal = BUY if score >= _BUY_THRESHOLD else (SELL if score <= _SELL_THRESHOLD else HOLD)

    # VIX gate: suppress BUY in high-fear environments ────────────────────────
    if signal == BUY and vix_level is not None and np.isfinite(vix_level) and vix_level >= _VIX_GATE:
        details["vix_gate_triggered"] = round(vix_level, 1)
        signal = HOLD

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

    # 3-month daily bars for prev_close, avg_volume, RSI, MA50
    prev_close = avg_volume = rsi = above_ma50 = None
    try:
        daily = _with_retry(lambda: t.history(period="3mo", interval="1d"), retries=3, base_delay=1.0)
        if daily is not None and not daily.empty and len(daily) >= 2:
            close      = daily["Close"]
            prev_close = float(close.iloc[-2])
            avg_volume = float(daily["Volume"].mean())
            rsi_s      = _compute_rsi(close)
            rsi_v      = float(rsi_s.iloc[-1])
            rsi        = rsi_v if np.isfinite(rsi_v) else None
            if len(close) >= _MA_WINDOW:
                ma50_v   = float(close.rolling(_MA_WINDOW).mean().iloc[-1])
                above_ma50 = float(close.iloc[-1]) > ma50_v
    except Exception as exc:
        logger.warning("compute_signal_single(%s): daily fetch failed: %s", ticker, exc)

    # Market context: VIX + sector ETF
    mkt_ctx   = _fetch_market_context()
    vix_level = mkt_ctx.get("vix")
    etf       = _sector_etf(ticker)
    sector_ret = mkt_ctx.get("etf_returns", {}).get(etf)

    # Analyst consensus
    analyst_data = _fetch_analyst_data(ticker)
    analyst_rec  = analyst_data.get("recommendation_mean")

    return _compute_signal(
        bars_10,
        prev_close       = prev_close,
        avg_daily_volume = avg_volume,
        macro_vote       = macro_vote,
        rsi              = rsi,
        above_ma50       = above_ma50,
        sector_ret       = sector_ret,
        vix_level        = vix_level,
        analyst_rec      = analyst_rec,
    )


def refresh_signals(tickers: list[str], macro_vote: int = 0) -> dict[str, dict]:
    """
    Download intraday + daily data for all *tickers* and compute signals.
    Returns {ticker: {signal, score, votes, details}}.
    """
    logger.info("Refreshing signals for %d tickers (macro_vote=%d)...", len(tickers), macro_vote)

    intraday    = _get_intraday_bars(tickers)
    daily       = _get_daily_info(tickers)
    mkt_ctx     = _fetch_market_context()
    analyst_map = _fetch_analyst_batch(tickers)

    vix_level    = mkt_ctx.get("vix")
    etf_returns  = mkt_ctx.get("etf_returns", {})

    signals: dict[str, dict] = {}
    for ticker in tickers:
        bars    = intraday.get(ticker)
        bars_10 = _extract_first_10_min(bars)
        d       = daily.get(ticker, {})
        etf     = _sector_etf(ticker)
        a       = analyst_map.get(ticker, {})
        signals[ticker] = _compute_signal(
            bars_10,
            prev_close       = d.get("prev_close"),
            avg_daily_volume = d.get("avg_volume"),
            macro_vote       = macro_vote,
            rsi              = d.get("rsi"),
            above_ma50       = d.get("above_ma50"),
            sector_ret       = etf_returns.get(etf),
            vix_level        = vix_level,
            analyst_rec      = a.get("recommendation_mean"),
        )

    buy_ct  = sum(1 for v in signals.values() if v["signal"] == BUY)
    sell_ct = sum(1 for v in signals.values() if v["signal"] == SELL)
    hold_ct = sum(1 for v in signals.values() if v["signal"] == HOLD)
    logger.info("Signals complete — BUY: %d  SELL: %d  HOLD: %d  VIX=%.1f",
                buy_ct, sell_ct, hold_ct, vix_level or 0)
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
