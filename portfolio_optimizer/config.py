"""Tunable constants for the optimizer."""

from pathlib import Path

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04  # ~4% annual; adjust to current 10Y treasury if you like

CACHE_DIR = Path(__file__).resolve().parent.parent / "ticker_cache"
CACHE_STALE_DAYS = 4  # tolerate up to 4-day gap (weekends + 1 holiday)
FUND_STALE_DAYS = 1  # fundamentals barely move intraday; refresh daily

# yfinance has no official rate limit but throttles aggressively under load.
# A small inter-request delay keeps a 40+ ticker fetch from getting throttled.
FETCH_DELAY_SECONDS = 1.0

FUND_FIELDS = (
    "trailingPE",
    "forwardPE",
    "pegRatio",
    "priceToBook",
    "dividendYield",
    "trailingEps",
    "forwardEps",
)
