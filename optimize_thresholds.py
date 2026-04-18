"""
optimize_thresholds.py  (v4 — 9-indicator long-only + analyst weight sweep)
----------------------------------------------------------------------------
Grid-search signal thresholds to maximise cumulative strategy return
minus buy-and-hold over trailing 1-year daily OHLCV bars.

Indicators in backtest (8 total — analyst treated as static constant):
  1. Gap          ±gap_t %
  2. Momentum     ±mom_t %
  3. VWAP proxy   ±vwap_t %
  4. Volume       vol_r × avg  (confirms momentum)
  5. RSI-14       > 50 = +1, < 50 = -1
  6. MA-50        close > MA50 = +1, else -1
  7. Sector ETF   ETF return > 0 = +1, < 0 = -1
  8. Analyst      current recommendationMean (static; ≤2.5=+1, ≥3.5=-1)
                  counted with analyst_weight multiplier (1×, 2×, 3×)

NOTE: analyst vote uses today's consensus as a constant across the backtest
window, which approximates reality (S&P 500 analyst ratings are sticky).
Live model also includes macro_trend (9th vote, not testable in daily OHLCV).

VIX gate: if VIX ≥ vix_gate → force HOLD regardless of score.
Strategy: BUY = long next bar, SELL/HOLD = flat (no shorts).
"""

import warnings
warnings.filterwarnings("ignore")

import time
from itertools import product
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

# ── Diversified S&P 500 sample ────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM",  "JNJ",  "XOM",  "BAC",  "UNH",  "V",    "HD",
    "PG",   "MA",   "ABBV", "CVX",  "LLY",  "MRK",
    "COST", "WMT",  "NFLX", "INTC", "AMD",
]

SECTOR_ETF = {
    **{t: "XLK" for t in ["AAPL","MSFT","NVDA","INTC","AMD"]},
    **{t: "XLY" for t in ["AMZN","TSLA","HD","COST"]},
    **{t: "XLC" for t in ["GOOGL","META","NFLX"]},
    **{t: "XLF" for t in ["JPM","BAC","V","MA"]},
    **{t: "XLV" for t in ["JNJ","UNH","ABBV","LLY","MRK"]},
    **{t: "XLE" for t in ["XOM","CVX"]},
    **{t: "XLP" for t in ["PG","WMT"]},
}
ALL_ETFS = list(set(SECTOR_ETF.values()))

VOL_WINDOW  = 20
RSI_WINDOW  = 14
MA_WINDOW   = 50
TARGET_DAYS = 252

# ── Parameter grid ────────────────────────────────────────────────────────────
GAP_THRESHOLDS    = [0.1, 0.25, 0.5, 0.75, 1.0]
MOM_THRESHOLDS    = [0.1, 0.2,  0.3, 0.5]
VWAP_THRESHOLDS   = [0.1, 0.2,  0.3]
VOL_RATIOS        = [1.0, 1.2,  1.5, 2.0]
BUY_THRESHOLDS    = [1, 2, 3, 4]
VIX_GATES         = [20.0, 25.0, 30.0, None]
ANALYST_WEIGHTS   = [0, 1, 2, 3]   # 0 = analyst vote disabled

MIN_BUY_SIGNALS = 10
MIN_STOCKS      = 12


def _compute_rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def download_stocks(tickers):
    print(f"Downloading 2y daily bars for {len(tickers)} tickers…")
    out = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period="2y", interval="1d", auto_adjust=True)
            if h is not None and not h.empty:
                h = h[["Open","High","Low","Close","Volume"]].dropna()
                if len(h) >= max(VOL_WINDOW, MA_WINDOW) + TARGET_DAYS + 2:
                    out[t] = h
                    print(f"  {t}: {len(h)} bars")
                else:
                    print(f"  {t}: too short ({len(h)})")
        except Exception as e:
            print(f"  {t}: FAILED — {e}")
    return out


def download_aux(fetch_period="2y"):
    """Download VIX and sector ETFs."""
    print("Downloading VIX and sector ETFs…")
    aux = {}
    for sym in ["^VIX"] + ALL_ETFS:
        try:
            h = yf.Ticker(sym).history(period=fetch_period, interval="1d", auto_adjust=True)
            if h is not None and not h.empty:
                aux[sym] = h["Close"].dropna()
        except Exception as e:
            print(f"  {sym}: FAILED — {e}")
    return aux


def fetch_analyst_votes(tickers):
    """
    Fetch current analyst recommendationMean for each ticker.
    Returns {ticker: vote} where vote ∈ {-1, 0, 1} based on rating.
    Uses today's consensus as a static proxy for the trailing year.
    """
    print("Fetching analyst consensus ratings…")
    votes = {}

    def _fetch_one(t):
        try:
            info = yf.Ticker(t).info
            rec  = info.get("recommendationMean")
            if rec is None or not np.isfinite(rec):
                return t, 0
            v = 1 if rec <= 2.5 else (-1 if rec >= 3.5 else 0)
            return t, v
        except Exception:
            return t, 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        for ticker, vote in pool.map(_fetch_one, tickers):
            votes[ticker] = vote
            rec_str = {1: "Buy (+1)", 0: "Hold (0)", -1: "Sell (-1)"}[vote]
            print(f"  {ticker}: {rec_str}")
    return votes


def precompute_features(datasets, aux, analyst_votes):
    """Pre-compute all indicator values for each stock's 1-year window."""
    features = {}
    vix_raw = aux.get("^VIX", pd.Series(dtype=float))
    vix_raw.index = vix_raw.index.normalize()

    for ticker, hist in datasets.items():
        window = hist.iloc[-(TARGET_DAYS + 1):]

        o  = window["Open"].values
        h  = window["High"].values
        l  = window["Low"].values
        c  = window["Close"].values
        v  = window["Volume"].values

        prev_c      = np.empty_like(c); prev_c[0] = np.nan; prev_c[1:] = c[:-1]
        gap_pct     = (o - prev_c) / prev_c * 100
        mom_pct     = (c - o) / o * 100
        typical     = (h + l + c) / 3
        vs_vwap     = (c - typical) / typical * 100

        avg_vol_full = (
            pd.Series(hist["Volume"].values)
            .rolling(VOL_WINDOW).mean().shift(1).values
        )
        avg_vol_w = avg_vol_full[-(TARGET_DAYS + 1):]
        vol_ratio = np.where(avg_vol_w > 0, v / avg_vol_w, np.nan)

        # RSI-14 (shifted 1 — no lookahead)
        rsi_full = _compute_rsi(hist["Close"]).shift(1)
        rsi_w    = rsi_full.values[-(TARGET_DAYS + 1):]

        # MA-50 vs close (shifted 1)
        ma50_full  = hist["Close"].rolling(MA_WINDOW).mean().shift(1)
        ma50_w     = ma50_full.values[-(TARGET_DAYS + 1):]
        above_ma50 = c > ma50_w

        # Sector ETF daily return
        etf_sym = SECTOR_ETF.get(ticker, "SPY")
        etf_raw = aux.get(etf_sym, pd.Series(dtype=float))
        etf_ret_full = etf_raw.pct_change()
        etf_ret_full.index = etf_ret_full.index.normalize()
        win_dates = window.index.normalize()
        etf_ret_w = etf_ret_full.reindex(win_dates).values

        # VIX level
        vix_w = vix_raw.reindex(win_dates).values

        # Forward return
        fwd      = np.empty_like(c)
        fwd[:-1] = c[1:] / c[:-1] - 1
        fwd[-1]  = np.nan

        # Static analyst vote (constant across all backtest days)
        analyst_v = analyst_votes.get(ticker, 0)

        # Strip warmup row
        features[ticker] = {
            "gap":        gap_pct[1:],
            "mom":        mom_pct[1:],
            "vwap":       vs_vwap[1:],
            "vol_ratio":  vol_ratio[1:],
            "rsi":        rsi_w[1:],
            "above_ma50": above_ma50[1:],
            "sector_ret": etf_ret_w[1:],
            "vix":        vix_w[1:],
            "analyst":    analyst_v,     # scalar ∈ {-1, 0, 1}
            "fwd":        fwd[1:],
            "close":      c,
        }
    return features


def eval_params(features, gap_t, mom_t, vwap_t, vol_r, buy_thr, vix_gate, analyst_w):
    sell_thr  = -buy_thr
    alphas, n_buy_list = [], []

    for ticker, f in features.items():
        fwd   = f["fwd"]
        valid = ~np.isnan(fwd)

        v_gap  = np.where(f["gap"]  >=  gap_t, 1, np.where(f["gap"]  <= -gap_t,  -1, 0))
        v_mom  = np.where(f["mom"]  >=  mom_t, 1, np.where(f["mom"]  <= -mom_t,  -1, 0))
        v_vwap = np.where(f["vwap"] >= vwap_t, 1, np.where(f["vwap"] <= -vwap_t, -1, 0))
        v_vol  = np.where(~np.isnan(f["vol_ratio"]) & (f["vol_ratio"] >= vol_r), v_mom, 0)

        rsi = f["rsi"]
        v_rsi = np.where(~np.isnan(rsi), np.where(rsi > 50, 1, -1), 0)

        v_ma50 = np.where(f["above_ma50"], 1, -1)

        sr = f["sector_ret"]
        v_sector = np.where(~np.isnan(sr), np.where(sr > 0, 1, np.where(sr < 0, -1, 0)), 0)

        # Analyst vote (static scalar, weighted)
        v_analyst = f["analyst"] * analyst_w

        score = v_gap + v_mom + v_vwap + v_vol + v_rsi + v_ma50 + v_sector + v_analyst
        buy_m = (score >= buy_thr) & valid

        # VIX gate
        if vix_gate is not None:
            vix = f["vix"]
            high_fear = ~np.isnan(vix) & (vix >= vix_gate)
            buy_m = buy_m & ~high_fear

        n_buy = int(buy_m.sum())
        if n_buy < MIN_BUY_SIGNALS:
            continue

        strat = np.where(buy_m, fwd, 0.0)
        strat = np.where(np.isnan(strat), 0.0, strat)
        cum_strat = float(np.prod(1 + strat) - 1)

        close = f["close"]
        bh    = float(close[-1] / close[1] - 1)

        alphas.append(cum_strat - bh)
        n_buy_list.append(n_buy)

    if len(alphas) < MIN_STOCKS:
        return None

    return {
        "avg_alpha":    float(np.mean(alphas)),
        "med_alpha":    float(np.median(alphas)),
        "pct_beats_bh": float(np.mean(np.array(alphas) > 0)),
        "avg_n_buy":    float(np.mean(n_buy_list)),
        "n_stocks":     len(alphas),
    }


def analyst_weight_summary(features, best_params, analyst_votes):
    """Compare analyst_weight 0–3 at the best non-analyst params."""
    gap_t, mom_t, vwap_t, vol_r, buy_thr, vix_gate = best_params
    print("\n" + "="*84)
    print("ANALYST WEIGHT SENSITIVITY  (best non-analyst thresholds held constant)")
    print(f"  gap={gap_t}  mom={mom_t}  vwap={vwap_t}  vol_r={vol_r}  buy_thr={buy_thr}  vix_gate={vix_gate}")
    print(f"  {'weight':>8}  {'avg_alpha':>12}  {'med_alpha':>12}  {'pct_beats_bh':>14}  {'avg_buys':>10}")
    for w in [0, 1, 2, 3]:
        r = eval_params(features, gap_t, mom_t, vwap_t, vol_r, buy_thr, vix_gate, w)
        if r:
            print(f"  {w:>8}  {r['avg_alpha']:+12.4f}  {r['med_alpha']:+12.4f}  {r['pct_beats_bh']:>14.1%}  {r['avg_n_buy']:>10.1f}")

    print("\nAnalyst vote breakdown across sample:")
    counts = {1: 0, 0: 0, -1: 0}
    for v in analyst_votes.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"  Buy (+1): {counts[1]}   Hold (0): {counts[0]}   Sell (-1): {counts[-1]}")


def main():
    t0 = time.time()

    datasets      = download_stocks(TICKERS)
    aux           = download_aux()
    analyst_votes = fetch_analyst_votes(list(datasets.keys()))
    print(f"\nLoaded {len(datasets)} stocks + {len(aux)} aux series in {time.time()-t0:.0f}s\n")

    print("Pre-computing features…")
    features = precompute_features(datasets, aux, analyst_votes)

    combos = list(product(
        GAP_THRESHOLDS, MOM_THRESHOLDS, VWAP_THRESHOLDS,
        VOL_RATIOS, BUY_THRESHOLDS, VIX_GATES, ANALYST_WEIGHTS,
    ))
    print(f"Grid-searching {len(combos)} combinations…\n")

    rows = []
    for i, (gap_t, mom_t, vwap_t, vol_r, buy_thr, vix_gate, analyst_w) in enumerate(combos):
        r = eval_params(features, gap_t, mom_t, vwap_t, vol_r, buy_thr, vix_gate, analyst_w)
        if r:
            rows.append(dict(
                gap_t=gap_t, mom_t=mom_t, vwap_t=vwap_t,
                vol_r=vol_r, buy_thr=buy_thr, sell_thr=-buy_thr,
                vix_gate=str(vix_gate), analyst_w=analyst_w, **r,
            ))
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(combos)}…")

    if not rows:
        print("No valid parameter sets.")
        return

    df = pd.DataFrame(rows).sort_values("avg_alpha", ascending=False)

    # ── Summary by analyst weight ─────────────────────────────────────────────
    print("\n" + "="*84)
    print("AVERAGE ALPHA BY ANALYST_WEIGHT  (across all valid param combos)")
    aw_group = df.groupby("analyst_w")["avg_alpha"].agg(["mean","median","count"])
    aw_group.columns = ["mean_alpha", "median_alpha", "n_combos"]
    print(aw_group.to_string())

    # ── Top 25 overall ────────────────────────────────────────────────────────
    print("\n" + "="*84)
    print("TOP 25 — 9-indicator long-only (sorted by avg alpha vs buy-and-hold)")
    print("="*84)
    cols = ["gap_t","mom_t","vwap_t","vol_r","buy_thr","vix_gate","analyst_w",
            "avg_alpha","med_alpha","pct_beats_bh","avg_n_buy"]
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda x: f"{x:+.4f}")
    print(df[cols].head(25).to_string(index=False))

    # ── Best per analyst_weight ───────────────────────────────────────────────
    print("\n" + "="*84)
    print("BEST PARAMS PER ANALYST WEIGHT")
    for w in ANALYST_WEIGHTS:
        sub = df[df["analyst_w"] == w]
        if sub.empty:
            continue
        b = sub.iloc[0]
        print(f"\n  analyst_w={w}: avg_alpha={b['avg_alpha']:+.4f}  pct_beats={b['pct_beats_bh']:.1%}")
        print(f"    gap={b['gap_t']}  mom={b['mom_t']}  vwap={b['vwap_t']}  "
              f"vol_r={b['vol_r']}  buy_thr={int(b['buy_thr'])}  vix_gate={b['vix_gate']}")

    best = df.iloc[0]

    # ── Analyst weight sensitivity at best base thresholds ───────────────────
    analyst_weight_summary(
        features,
        (best["gap_t"], best["mom_t"], best["vwap_t"],
         best["vol_r"], int(best["buy_thr"]), None if best["vix_gate"] == "None" else float(best["vix_gate"])),
        analyst_votes,
    )

    print("\n" + "="*84)
    print("RECOMMENDED PARAMETERS (update fetcher.py + backtest.py):")
    print(f"  _GAP_PCT_THRESHOLD      = {best['gap_t']}")
    print(f"  _MOMENTUM_PCT_THRESHOLD = {best['mom_t']}")
    print(f"  _VWAP_PCT_THRESHOLD     = {best['vwap_t']}")
    print(f"  _VOL_HIGH_RATIO         = {best['vol_r']}")
    print(f"  _BUY_THRESHOLD          = {int(best['buy_thr'])}")
    print(f"  _SELL_THRESHOLD         = {int(best['sell_thr'])}")
    print(f"  _VIX_GATE               = {best['vix_gate']}")
    print(f"  analyst_weight          = {int(best['analyst_w'])}")
    print(f"\n  avg alpha vs B&H : {best['avg_alpha']:+.4f}")
    print(f"  pct beats B&H    : {best['pct_beats_bh']:.1%}")
    print(f"  avg BUY days/yr  : {best['avg_n_buy']:.1f}")
    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
