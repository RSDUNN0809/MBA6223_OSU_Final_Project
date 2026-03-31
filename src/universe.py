"""
universe.py — S&P 500 constituent list.

Primary source: Wikipedia table (scraped via pandas).
Fallback: 30-stock hardcoded list used when network is unavailable.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── Fallback ticker list (used when Wikipedia scrape fails) ───────────────────
_FALLBACK = [
    ("AAPL", "Apple Inc.", "Technology"),
    ("MSFT", "Microsoft Corp.", "Technology"),
    ("NVDA", "NVIDIA Corp.", "Technology"),
    ("GOOGL", "Alphabet Inc.", "Communication Services"),
    ("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
    ("META", "Meta Platforms Inc.", "Communication Services"),
    ("TSLA", "Tesla Inc.", "Consumer Discretionary"),
    ("BRK-B", "Berkshire Hathaway", "Financials"),
    ("UNH", "UnitedHealth Group", "Health Care"),
    ("LLY", "Eli Lilly & Co.", "Health Care"),
    ("JPM", "JPMorgan Chase & Co.", "Financials"),
    ("V", "Visa Inc.", "Financials"),
    ("XOM", "Exxon Mobil Corp.", "Energy"),
    ("MA", "Mastercard Inc.", "Financials"),
    ("PG", "Procter & Gamble Co.", "Consumer Staples"),
    ("JNJ", "Johnson & Johnson", "Health Care"),
    ("HD", "Home Depot Inc.", "Consumer Discretionary"),
    ("COST", "Costco Wholesale", "Consumer Staples"),
    ("MRK", "Merck & Co. Inc.", "Health Care"),
    ("AVGO", "Broadcom Inc.", "Technology"),
    ("CVX", "Chevron Corp.", "Energy"),
    ("CRM", "Salesforce Inc.", "Technology"),
    ("BAC", "Bank of America Corp.", "Financials"),
    ("ABBV", "AbbVie Inc.", "Health Care"),
    ("NFLX", "Netflix Inc.", "Communication Services"),
    ("KO", "Coca-Cola Co.", "Consumer Staples"),
    ("PEP", "PepsiCo Inc.", "Consumer Staples"),
    ("WMT", "Walmart Inc.", "Consumer Staples"),
    ("TMO", "Thermo Fisher Scientific", "Health Care"),
    ("AMD", "Advanced Micro Devices", "Technology"),
]


def get_sp500() -> pd.DataFrame:
    """
    Return a DataFrame of S&P 500 constituents with columns:
        ticker, company, sector

    Scrapes the Wikipedia list; falls back to _FALLBACK on any error.
    """
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
        df.columns = ["ticker", "company", "sector"]
        # Wikipedia uses dots in some tickers (e.g. BRK.B) — yfinance uses hyphens
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        df = df.dropna(subset=["ticker"]).reset_index(drop=True)
        logger.info("S&P 500 universe loaded from Wikipedia: %d tickers.", len(df))
        return df
    except Exception as exc:
        logger.warning("Wikipedia scrape failed (%s) — using fallback list.", exc)
        return pd.DataFrame(_FALLBACK, columns=["ticker", "company", "sector"])
