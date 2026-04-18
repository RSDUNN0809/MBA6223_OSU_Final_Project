"""
app.py — Longshot Stock Prediction (LSP) Flask application.

Routes
------
GET  /                          Single-page dashboard
GET  /api/universe              S&P 500 ticker list (cached daily)
GET  /api/trends                Macro proxy scores (cached daily)
GET  /api/stock/<ticker>        Live signal + metrics (computed on demand, cached per ticker per day)
GET  /api/stock/<ticker>/history  5-day signal-vs-outcome lookback
GET  /api/refresh               Manual refresh — clears all caches so the next request fetches live data
GET  /api/status                Health / cache status

Finance-depth layer
-------------------
GET  /api/alpha                 12-month trailing-alpha ranking (top/bottom 10)
GET  /api/composite/<ticker>    Three-pillar composite score
GET  /api/backtest/<ticker>     Backtest results (1mo / 1y / 5y)
GET  /api/backtest/<ticker>/<window>  Single-window backtest
GET  /drilldown/<ticker>        Per-stock drill-down page
GET  /backtest                  Backtest dashboard page
GET  /alpha                     Alpha ranking page

Signal computation
------------------
Signals are computed ON DEMAND when a stock is selected — no scheduled batch.
The first request for a ticker each day fetches live 1-minute intraday bars
and computes the BUY/HOLD/SELL signal in real time (~2–4 s).  The result is
cached in memory for the rest of the day.  Clicking Refresh clears the cache
so the next lookup re-fetches live data.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Any

import pytz
from flask import Flask, jsonify, render_template, request

from src.fetcher import compute_signal_single, get_stock_info, get_weekly_history
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
_TRENDS_FILE   = os.path.join(DATA_DIR, "trends_cache.json")
_REFRESH_FILE  = os.path.join(DATA_DIR, "last_refresh.json")
_DEPTH_FILE    = os.path.join(DATA_DIR, "depth_cache.json")
_BACKTEST_FILE = os.path.join(DATA_DIR, "backtest_cache.json")

# ── In-memory cache ────────────────────────────────────────────────────────────
# Signals are now per-ticker, per-day dicts: {"_date": str, signal, score, ...}
_mem: dict[str, Any] = {
    "universe": None,   # list[dict]
    "signals": {},      # {ticker: {"_date": str, signal, score, votes, details}}
    "metrics": {},      # {ticker: {"_date": str, ...fields}}
    "history": {},      # {ticker: {"_date": str, data: list[dict]}}
    "trends": None,     # {data: list[dict], fetched_at: str}
    "last_refresh": None,
    "depth": None,      # PipelineResult dict
    "backtests": {},    # {ticker: {"_date": str, data: dict}}
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


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _clear_all_caches():
    """Wipe all per-ticker in-memory caches so the next request fetches live data."""
    _mem["signals"]  = {}
    _mem["metrics"]  = {}
    _mem["history"]  = {}
    _mem["backtests"] = {}
    _mem["trends"]   = None


# ── Startup — warm universe + depth from file ──────────────────────────────────
def _warm_cache():
    universe = _load(_UNIVERSE_FILE, [])
    if universe:
        _mem["universe"] = universe

    trends_payload = _load(_TRENDS_FILE, {})
    today = date.today().isoformat()
    if trends_payload.get("data") and trends_payload.get("fetched_at", "")[:10] == today:
        _mem["trends"] = trends_payload

    depth_cached = _load(_DEPTH_FILE, {})
    if depth_cached:
        _mem["depth"] = depth_cached

    refresh_info = _load(_REFRESH_FILE, {})
    if refresh_info.get("last_refresh"):
        _mem["last_refresh"] = refresh_info["last_refresh"]

    logger.info(
        "Cache warmed — universe: %d, trends: %s, depth: %s",
        len(_mem["universe"] or []),
        "yes" if _mem["trends"] else "no",
        "yes" if _mem["depth"] else "no",
    )


_warm_cache()


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
    return jsonify(_get_universe())


@app.route("/api/trends")
def api_trends():
    """Return the Google Trends macro scores."""
    return jsonify(_get_trends())


@app.route("/api/stock/<ticker>")
def api_stock(ticker: str):
    """
    Return live financial metrics + on-demand BUY/HOLD/SELL signal for *ticker*.

    Signal: computed fresh from today's 1-min intraday bars the first time a
    ticker is requested each day, then cached in memory.  Clicking Refresh
    clears the cache so the next call re-fetches live data.

    Metrics (price, P/E, etc.): same lazy per-ticker per-day cache.
    """
    ticker = ticker.upper()
    today_str = date.today().isoformat()

    # ── Signal ────────────────────────────────────────────────────────────────
    sig_cached = _mem["signals"].get(ticker, {})
    if sig_cached.get("_date") != today_str:
        # Derive current macro vote from cached trends (fast, already fetched)
        macro_vote = 0
        trends = _mem.get("trends") or {}
        if trends.get("data"):
            macro_vote = compute_macro_vote(trends["data"])

        logger.info("Computing live signal for %s (macro_vote=%+d)", ticker, macro_vote)
        result = compute_signal_single(ticker, macro_vote=macro_vote)
        result["_date"] = today_str
        _mem["signals"][ticker] = result
        sig_cached = result

    signal_data = sig_cached

    # ── Metrics ───────────────────────────────────────────────────────────────
    cached_metrics = _mem["metrics"].get(ticker, {})
    if cached_metrics.get("_date") != today_str:
        info = get_stock_info(ticker)
        cached_metrics = {"_date": today_str, **info}
        _mem["metrics"][ticker] = cached_metrics

    # ── Composite (from depth cache, optional) ────────────────────────────────
    composite_block = None
    depth = _get_depth()
    if depth and isinstance(depth.get("scored"), dict):
        composite_block = depth["scored"].get(ticker)

    # ── Hit rate (from backtest cache if already computed) ────────────────────
    hit_rate = None
    bt_cached = _mem["backtests"].get(ticker, {})
    if bt_cached.get("data"):
        hit_rate = bt_cached["data"].get("1y", {}).get("hit_rate")

    prev_close = cached_metrics.get("previous_close")

    return jsonify({
        "ticker": ticker,
        "signal": signal_data.get("signal", "HOLD"),
        "score": signal_data.get("score", 0),
        "votes": signal_data.get("votes", {}),
        "details": signal_data.get("details", {}),
        "prev_close": prev_close,
        "hit_rate": hit_rate,
        **{k: v for k, v in cached_metrics.items() if k not in ("_date",)},
        "composite": composite_block,
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
    if cached.get("_date") == today_str:
        return jsonify({"ticker": ticker, "history": cached["data"]})

    macro_vote = 0
    trends = _mem.get("trends") or {}
    if trends.get("data"):
        macro_vote = compute_macro_vote(trends["data"])

    history = get_weekly_history(ticker, macro_vote=macro_vote)
    _mem["history"][ticker] = {"_date": today_str, "data": history}
    return jsonify({"ticker": ticker, "history": history})


@app.route("/api/refresh", methods=["POST", "GET"])
def api_refresh():
    """
    Clear all per-ticker caches so the next stock request fetches live data.
    Completes instantly — no background job.  The signal for the displayed
    stock is recomputed the moment the frontend re-fetches it.
    """
    _clear_all_caches()
    now_et = datetime.now(ET)
    refresh_ts = now_et.isoformat()
    _save(_REFRESH_FILE, {"last_refresh": refresh_ts})
    _mem["last_refresh"] = refresh_ts
    logger.info("Manual cache clear at %s", refresh_ts)
    return jsonify({"status": "cache_cleared", "cleared_at": refresh_ts})


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
    if cached.get("_date") == today_str:
        return jsonify({"ticker": ticker, **cached["data"]})

    try:
        result = depth_backtest.run_multi_window_backtest(ticker)
    except Exception as exc:
        logger.exception("backtest(%s) failed", ticker)
        return jsonify({"ticker": ticker, "error": str(exc)}), 500

    _mem["backtests"][ticker] = {"_date": today_str, "data": result}
    try:
        all_bt = _load(_BACKTEST_FILE, {})
        all_bt[ticker] = {"_date": today_str, "data": result}
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


@app.route("/alpha")
def alpha_page():
    """Alpha ranking dashboard page."""
    return render_template("alpha.html")


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
