"""Raw yfinance wrappers and post-fetch data-quality checks."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from . import cache
from .config import FUND_FIELDS


def yf_close(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch adjusted closes for one ticker. Returns empty Series on any failure."""
    try:
        raw = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False,
        )
    except Exception as e:
        print(f"    ! yfinance error for {ticker}: {type(e).__name__}: {e}")
        return pd.Series(dtype=float, name="close")

    if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
        return pd.Series(dtype=float, name="close")

    close = raw["Close"]
    # Newer yfinance returns MultiIndex columns even for a single ticker;
    # raw["Close"] is then a 1-col DataFrame rather than a Series.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.name = "close"
    return close


def yf_fundamentals(ticker: str) -> dict:
    """Pull a small whitelist of valuation fields from yfinance. Empty dict on failure."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        print(f"    ! fundamentals error for {ticker}: {type(e).__name__}: {e}")
        return {}
    return {k: info.get(k) for k in FUND_FIELDS if info.get(k) is not None}


def looks_clean(close: pd.Series, ticker: str) -> bool:
    """Reject series with non-positive prices or absurd single-day jumps.

    yfinance occasionally returns data spliced across reused symbols — e.g. an
    old delisted security and a newer one sharing the same ticker — which
    produces multi-thousand-percent daily moves that wreck the optimizer.
    """
    if close.empty or len(close) < 2:
        print(f"    ! {ticker}: too few points ({len(close)}) — dropping")
        return False
    if (close <= 0).any():
        print(f"    ! {ticker}: contains non-positive prices — dropping")
        return False
    rets = close.pct_change().dropna()
    extreme = rets[rets.abs() > 2.0]  # >200% in one day → almost certainly bad
    if not extreme.empty:
        worst = extreme.iloc[extreme.abs().argmax()]
        when = extreme.index[extreme.abs().argmax()].date()
        print(
            f"    ! {ticker}: {len(extreme)} day(s) with |return|>200% "
            f"(worst {worst:+.0%} on {when}) — likely spliced data, dropping. "
            f"Delete {cache.price_path(ticker).name} to retry fetch."
        )
        return False
    return True
