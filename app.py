"""
app.py — Longshot Stock Prediction (LSP) Flask application.

Routes
------
GET  /                       Serve the single-page UI
GET  /api/universe            S&P 500 ticker list (cached)
GET  /api/trends              Google Trends macro scores (cached)
GET  /api/stock/<ticker>      Financial metrics + Buy/Hold/Sell signal
POST /api/refresh             Manual full refresh (requires X-Refresh-Secret header)

Daily refresh
-------------
APScheduler fires _daily_refresh() at 9:40 AM ET on weekdays.
A /api/refresh POST endpoint allows external cron services (e.g. cron-job.org)
to trigger the same refresh — useful when the free-tier dyno has been sleeping.
"""
from __future__ import annotations

import json
import logging
import os
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

# ── In-memory cache (fast path; file cache survives across cold restarts) ─────
_mem: dict[str, Any] = {
    "universe": None,    # list[dict]
    "signals": {},       # {ticker: {signal, score, votes, details}}
    "metrics": {},       # {ticker: {date, ...fields}}
    "history": {},       # {ticker: {date: str, data: list[dict]}}
    "trends": None,      # {data: list[dict], fetched_at: str}
    "last_refresh": None,
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


# ── Daily refresh ──────────────────────────────────────────────────────────────

def _daily_refresh():
    """
    Run the full daily refresh:
    1. Refresh Google Trends + compute macro vote
    2. Re-scrape S&P 500 universe
    3. Batch-download intraday data, compute signals for all tickers
    4. Persist all caches
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

    # Step 4: Metrics and history caches are per-ticker/day; clear so fresh
    # fetches happen on the next user request.
    _mem["metrics"] = {}
    _mem["history"] = {}

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

    logger.info(
        "Cache warmed — universe: %d, signals: %d, trends: %s",
        len(_mem["universe"] or []),
        len(_mem["signals"]),
        "yes" if _mem["trends"] else "no",
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
        import threading
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

    return jsonify({
        "ticker": ticker,
        "signal": signal_data.get("signal", "HOLD"),
        "score": signal_data.get("score", 0),
        "votes": signal_data.get("votes", {}),
        **{k: v for k, v in cached_metrics.items() if k != "date"},
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

    import threading
    threading.Thread(target=_daily_refresh, daemon=True).start()
    return jsonify({"status": "refresh_started", "triggered_at": datetime.now(ET).isoformat()})


@app.route("/api/status")
def api_status():
    """Health / status endpoint — useful for uptime monitors."""
    return jsonify({
        "status": "ok",
        "last_refresh": _mem.get("last_refresh"),
        "universe_count": len(_mem["universe"] or []),
        "signals_count": len(_mem["signals"]),
        "trends_available": _mem["trends"] is not None,
        "server_time_et": datetime.now(ET).isoformat(),
    })


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
