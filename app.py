"""
app.py — Longshot Stock Prediction (LSP) Flask application.

Routes (original)
-----------------
GET  /                          Serve the single-page UI
GET  /api/universe              S&P 500 ticker list (cached)
GET  /api/trends                Google Trends macro scores (cached)
GET  /api/stock/<ticker>        Financial metrics + Buy/Hold/Sell signal
POST /api/refresh               Manual full refresh (requires X-Refresh-Secret header)

Routes (finance-depth layer)
-----------------------------
GET  /api/alpha                 12-month trailing-alpha ranking (top 10 + bottom 10)
GET  /api/composite/<ticker>    Three-pillar composite score for one stock
GET  /api/backtest/<ticker>     Backtest results (1mo / 1y / 5y)
GET  /api/backtest/<ticker>/<window>  Single-window backtest
GET  /drilldown/<ticker>        Per-stock metric drill-down page
GET  /backtest                  Backtest dashboard page

Daily refresh
-------------
APScheduler fires _daily_refresh() at 9:40 AM ET on weekdays. That job now also
runs the finance-depth pipeline (alpha ranking -> top/bottom 10 -> composite
scoring) and caches the result to data/depth_cache.json.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date, datetime
from functools import wraps
from typing import Any

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask import Flask, abort, jsonify, render_template, request

from src.fetcher import get_stock_info, get_weekly_history, refresh_signals
from src.trends import compute_macro_vote, get_macro_trends
from src.universe import get_sp500

# finance-depth package
from finance_depth import signals as depth_signals
from finance_depth import backtest as depth_backtest
from finance_depth.sentiment_modifier import compute_trend_sentiment

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

_UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.json")
_SIGNALS_FILE = os.path.join(DATA_DIR, "signals_cache.json")
_METRICS_FILE = os.path.join(DATA_DIR, "metrics_cache.json")
_TRENDS_FILE = os.path.join(DATA_DIR, "trends_cache.json")
_REFRESH_FILE = os.path.join(DATA_DIR, "last_refresh.json")
_DEPTH_FILE = os.path.join(DATA_DIR, "depth_cache.json")
_BACKTEST_FILE = os.path.join(DATA_DIR, "backtest_cache.json")

# ── In-memory cache (fast path; file cache survives across cold restarts) ─────
_mem: dict[str, Any] = {
    "universe": None,    # list[dict]
    "signals": {},       # {ticker: {signal, score, votes, details}}
    "metrics": {},       # {ticker: {date, ...fields}}
    "history": {},       # {ticker: {date: str, data: list[dict]}}
    "trends": None,      # {data: list[dict], fetched_at: str}
    "last_refresh": None,
    "depth": None,       # PipelineResult dict
    "backtests": {},     # {ticker: {date, data}}
}


# ── JSON helpers ───────────────────────────────────────────────────────────────

def _load(path: str, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}


def _save(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f)


# ── Universe ───────────────────────────────────────────────────────────────────

def _get_universe() -> list[dict]:
    """Return S&P 500 list — from memory, then file, then live fetch."""
    if _mem["universe"]:
        return _mem["universe"]
    cached = _load(_UNIVERSE_FILE, [])
    if cached:
        _mem["universe"] = cached
        return cached
    df = get_sp500()
    data = df.to_dict("records")
    _save(_UNIVERSE_FILE, data)
    _mem["universe"] = data
    return data


# ── Trends ─────────────────────────────────────────────────────────────────────

def _get_trends() -> dict:
    """Return macro trends — from memory, then file (today), then live fetch."""
    if _mem["trends"]:
        return _mem["trends"]

    cached = _load(_TRENDS_FILE, {})
    if cached.get("data") and cached.get("fetched_at", "")[:10] == date.today().isoformat():
        _mem["trends"] = cached
        return cached

    # Fetch fresh
    trends_list = get_macro_trends()
    result = {"data": trends_list, "fetched_at": datetime.now(ET).isoformat()}
    _save(_TRENDS_FILE, result)
    _mem["trends"] = result
    return result


# ── Finance-depth cache accessor ───────────────────────────────────────────────

def _get_depth() -> dict | None:
    """Return the cached three-pillar pipeline result, or None if not computed."""
    if _mem["depth"]:
        return _mem["depth"]
    cached = _load(_DEPTH_FILE, {})
    if cached:
        _mem["depth"] = cached
    return cached or None


# ── Daily refresh ──────────────────────────────────────────────────────────────

def _daily_refresh():
    """
    Run the full daily refresh:
    1. Refresh Google Trends + compute macro vote
    2. Re-scrape S&P 500 universe
    3. Batch-download intraday data, compute signals for all tickers
    4. Run finance-depth pipeline (alpha rank -> top/bottom 10 -> composite)
    5. Persist all caches
    """
    logger.info("=== Daily refresh starting ===")
    now_et = datetime.now(ET)

    # Step 1: Trends
    try:
        trends_list = get_macro_trends()
        macro_vote = compute_macro_vote(trends_list)
        trends_payload = {"data": trends_list, "fetched_at": now_et.isoformat()}
        _save(_TRENDS_FILE, trends_payload)
        _mem["trends"] = trends_payload
        logger.info("Trends refreshed. Macro vote: %+d", macro_vote)
    except Exception as exc:
        logger.warning("Trends refresh failed: %s — using vote=0", exc)
        macro_vote = 0

    # Step 2: Universe
    try:
        df = get_sp500()
        universe_data = df.to_dict("records")
        _save(_UNIVERSE_FILE, universe_data)
        _mem["universe"] = universe_data
        logger.info("Universe refreshed: %d tickers", len(universe_data))
    except Exception as exc:
        logger.warning("Universe refresh failed: %s — using cached list", exc)
        universe_data = _mem["universe"] or _load(_UNIVERSE_FILE, [])

    # Step 3: Signals
    tickers = [s["ticker"] for s in universe_data]
    try:
        signals = refresh_signals(tickers, macro_vote=macro_vote)
        signals_payload = {"data": signals, "fetched_at": now_et.isoformat()}
        _save(_SIGNALS_FILE, signals_payload)
        _mem["signals"] = signals
        logger.info("Signals refreshed for %d tickers.", len(signals))
    except Exception as exc:
        logger.error("Signal refresh failed: %s", exc)

    # Step 4: Finance-depth pipeline
    try:
        trend = compute_trend_sentiment()
        ticker_sector_map = {
            row["ticker"]: row.get("sector")
            for row in universe_data
            if row.get("ticker")
        }
        depth_result = depth_signals.run_full_pipeline(
            tickers,
            benchmark="^GSPC",
            top_n=10,
            bottom_n=10,
            trend=trend,
            ticker_sector_map=ticker_sector_map,
        ).to_dict()
        _save(_DEPTH_FILE, depth_result)
        _mem["depth"] = depth_result
        logger.info(
            "Depth pipeline: %d top, %d bottom scored; trend shift = %+.0f",
            len(depth_result.get("alpha_top", [])),
            len(depth_result.get("alpha_bottom", [])),
            depth_result.get("trend", {}).get("threshold_shift", 0.0),
        )
    except Exception as exc:
        logger.error("Depth pipeline failed: %s", exc)

    # Step 5: Clear per-ticker caches so fresh fetches happen on next request
    _mem["metrics"] = {}
    _mem["history"] = {}
    _mem["backtests"] = {}

    refresh_ts = now_et.isoformat()
    _save(_REFRESH_FILE, {"last_refresh": refresh_ts})
    _mem["last_refresh"] = refresh_ts

    logger.info("=== Daily refresh complete at %s ===", refresh_ts)


def _should_auto_refresh() -> bool:
    """
    Return True if today's data hasn't been computed yet and the market has
    been open for at least 10 minutes.
    """
    cached = _load(_REFRESH_FILE, {})
    last = cached.get("last_refresh") or _mem.get("last_refresh")
    if not last:
        return True
    last_dt = datetime.fromisoformat(last).astimezone(ET)
    now_et = datetime.now(ET)
    market_ready = now_et.hour * 60 + now_et.minute >= 9 * 60 + 40
    is_weekday = now_et.weekday() < 5
    return is_weekday and market_ready and last_dt.date() < now_et.date()


# ── Startup ────────────────────────────────────────────────────────────────────
# Load from file cache into memory on startup so the first request is fast.
def _warm_cache():
    universe = _load(_UNIVERSE_FILE, [])
    if universe:
        _mem["universe"] = universe

    signals_payload = _load(_SIGNALS_FILE, {})
    if signals_payload.get("data"):
        _mem["signals"] = signals_payload["data"]

    trends_payload = _load(_TRENDS_FILE, {})
    if trends_payload.get("data"):
        _mem["trends"] = trends_payload

    refresh_info = _load(_REFRESH_FILE, {})
    if refresh_info.get("last_refresh"):
        _mem["last_refresh"] = refresh_info["last_refresh"]

    depth_cached = _load(_DEPTH_FILE, {})
    if depth_cached:
        _mem["depth"] = depth_cached

    backtest_cached = _load(_BACKTEST_FILE, {})
    if backtest_cached:
        _mem["backtests"] = backtest_cached

    logger.info(
        "Cache warmed — universe: %d, signals: %d, trends: %s, depth: %s",
        len(_mem["universe"] or []),
        len(_mem["signals"]),
        "yes" if _mem["trends"] else "no",
        "yes" if _mem["depth"] else "no",
    )


_warm_cache()

# ── APScheduler — fires at 9:40 AM ET, Monday–Friday ──────────────────────────
_scheduler = BackgroundScheduler(timezone=ET)
_scheduler.add_job(
    _daily_refresh,
    CronTrigger(hour=9, minute=40, day_of_week="mon-fri", timezone=ET),
    id="daily_refresh",
    name="Morning signal refresh",
    misfire_grace_time=300,  # allow up to 5 min late if dyno was sleeping
)
_scheduler.start()
logger.info("APScheduler started — next refresh at 09:40 AM ET on weekdays.")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page dashboard."""
    last = _mem.get("last_refresh")
    if last:
        last_dt = datetime.fromisoformat(last).astimezone(ET)
        last_str = last_dt.strftime("%b %-d, %Y — %-I:%M %p ET")
    else:
        last_str = "Not yet refreshed"
    return render_template("index.html", last_refresh=last_str)


@app.route("/api/universe")
def api_universe():
    """Return the S&P 500 ticker list as JSON."""
    # Trigger auto-refresh check on first real request of the day
    if _should_auto_refresh():
        logger.info("Auto-refresh triggered via /api/universe request.")
        threading.Thread(target=_daily_refresh, daemon=True).start()
    return jsonify(_get_universe())


@app.route("/api/trends")
def api_trends():
    """Return the Google Trends macro scores."""
    return jsonify(_get_trends())


@app.route("/api/stock/<ticker>")
def api_stock(ticker: str):
    """
    Return financial metrics + Buy/Hold/Sell signal for *ticker*.
    Financial metrics are lazily fetched and cached per ticker per day.
    """
    ticker = ticker.upper()

    # Signal from morning cache
    signal_data = _mem["signals"].get(ticker) or {}

    # Financial metrics — check in-memory cache first
    today_str = date.today().isoformat()
    cached_metrics = _mem["metrics"].get(ticker, {})

    if cached_metrics.get("date") != today_str:
        # Fetch fresh from yfinance
        info = get_stock_info(ticker)
        cached_metrics = {"date": today_str, **info}
        _mem["metrics"][ticker] = cached_metrics
        # Persist to file (best-effort — don't crash if write fails)
        try:
            all_metrics = _load(_METRICS_FILE, {})
            all_metrics[ticker] = cached_metrics
            _save(_METRICS_FILE, all_metrics)
        except Exception:
            pass

    # Attach composite result if available for this ticker
    composite_block = None
    depth = _get_depth()
    if depth and isinstance(depth.get("scored"), dict):
        composite_block = depth["scored"].get(ticker)

    return jsonify({
        "ticker": ticker,
        "signal": signal_data.get("signal", "HOLD"),
        "score": signal_data.get("score", 0),
        "votes": signal_data.get("votes", {}),
        **{k: v for k, v in cached_metrics.items() if k != "date"},
        "composite": composite_block,   # None if ticker not in top/bottom 20
    })


@app.route("/api/stock/<ticker>/history")
def api_stock_history(ticker: str):
    """
    Return the last 5 completed trading days of signal-vs-outcome data.
    Each row tells you what signal the model would have given at 9:40 AM ET
    and whether following that signal would have been profitable by close.
    Results are cached per ticker per calendar day.
    """
    ticker = ticker.upper()

    today_str = date.today().isoformat()
    cached = _mem["history"].get(ticker, {})
    if cached.get("date") == today_str:
        return jsonify({"ticker": ticker, "history": cached["data"]})

    # Derive the current macro vote from cached trends
    macro_vote = 0
    if _mem.get("trends") and _mem["trends"].get("data"):
        from src.trends import compute_macro_vote
        macro_vote = compute_macro_vote(_mem["trends"]["data"])

    history = get_weekly_history(ticker, macro_vote=macro_vote)
    _mem["history"][ticker] = {"date": today_str, "data": history}
    return jsonify({"ticker": ticker, "history": history})


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """
    Trigger a manual full refresh.
    Protect with the REFRESH_SECRET environment variable.
    Set X-Refresh-Secret: <value> in the request header.
    If REFRESH_SECRET is not set, the endpoint is unprotected (dev mode).
    """
    secret = os.environ.get("REFRESH_SECRET", "")
    if secret and request.headers.get("X-Refresh-Secret") != secret:
        abort(403)

    threading.Thread(target=_daily_refresh, daemon=True).start()
    return jsonify({"status": "refresh_started", "triggered_at": datetime.now(ET).isoformat()})


@app.route("/api/status")
def api_status():
    """Health / status endpoint — useful for uptime monitors."""
    depth = _get_depth() or {}
    return jsonify({
        "status": "ok",
        "last_refresh": _mem.get("last_refresh"),
        "universe_count": len(_mem["universe"] or []),
        "signals_count": len(_mem["signals"]),
        "trends_available": _mem["trends"] is not None,
        "depth_available": bool(depth),
        "depth_generated_at": depth.get("generated_at"),
        "trend_shift": (depth.get("trend") or {}).get("threshold_shift"),
        "server_time_et": datetime.now(ET).isoformat(),
    })


# ── Routes (finance-depth layer) ───────────────────────────────────────────────

@app.route("/api/alpha")
def api_alpha():
    """Return the 12-month trailing-alpha ranking (top 10 + bottom 10)."""
    depth = _get_depth()
    if not depth:
        return jsonify({
            "error": "depth_not_computed_yet",
            "hint": "POST /api/refresh or wait for 9:40 AM ET scheduled refresh.",
        }), 503
    return jsonify({
        "generated_at": depth.get("generated_at"),
        "benchmark": depth.get("benchmark"),
        "trend": depth.get("trend"),
        "top": depth.get("alpha_top", []),
        "bottom": depth.get("alpha_bottom", []),
    })


@app.route("/api/composite/<ticker>")
def api_composite(ticker: str):
    """Three-pillar composite breakdown for a single stock."""
    ticker = ticker.upper()
    depth = _get_depth() or {}
    scored = (depth.get("scored") or {}).get(ticker)
    if scored:
        return jsonify({"ticker": ticker, "cached": True, **scored})

    # Not in the top/bottom 20 — compute on demand
    try:
        result = depth_signals.generate_signal(ticker)
        return jsonify({"ticker": ticker, "cached": False, **result})
    except Exception as exc:
        logger.exception("composite on-demand failed for %s", ticker)
        return jsonify({"ticker": ticker, "error": str(exc)}), 500


@app.route("/api/backtest/<ticker>")
def api_backtest_all(ticker: str):
    """Run 1-month, 1-year, 5-year backtests. Cached per-ticker per-day."""
    ticker = ticker.upper()
    today_str = date.today().isoformat()
    cached = _mem["backtests"].get(ticker, {})
    if cached.get("date") == today_str:
        return jsonify({"ticker": ticker, **cached["data"]})

    try:
        result = depth_backtest.run_multi_window_backtest(ticker)
    except Exception as exc:
        logger.exception("backtest(%s) failed", ticker)
        return jsonify({"ticker": ticker, "error": str(exc)}), 500

    _mem["backtests"][ticker] = {"date": today_str, "data": result}
    try:
        all_bt = _load(_BACKTEST_FILE, {})
        all_bt[ticker] = {"date": today_str, "data": result}
        _save(_BACKTEST_FILE, all_bt)
    except Exception:
        pass
    return jsonify({"ticker": ticker, **result})


@app.route("/api/backtest/<ticker>/<window>")
def api_backtest_window(ticker: str, window: str):
    """Run a single-window backtest (1mo / 1y / 5y)."""
    ticker = ticker.upper()
    window = window.lower()
    if window not in ("1mo", "1y", "5y"):
        return jsonify({"error": "window must be 1mo, 1y, or 5y"}), 400
    try:
        result = depth_backtest.run_backtest(ticker, window=window)
    except Exception as exc:
        logger.exception("backtest(%s, %s) failed", ticker, window)
        return jsonify({"ticker": ticker, "error": str(exc)}), 500
    return jsonify(result)


@app.route("/drilldown/<ticker>")
def drilldown(ticker: str):
    """Per-stock drill-down page showing all three pillars."""
    return render_template("drilldown.html", ticker=ticker.upper())


@app.route("/backtest")
def backtest_page():
    """Top-level backtest dashboard page."""
    return render_template("backtest.html")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
