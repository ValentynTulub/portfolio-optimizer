"""High-level data loading: combines cache reads with yfinance fetches."""

from __future__ import annotations

import time

import pandas as pd

from . import cache, fetch
from .config import CACHE_STALE_DAYS, FETCH_DELAY_SECONDS, FUND_STALE_DAYS


def fetch_prices(tickers: list[str], years: int) -> pd.DataFrame:
    """Return adjusted close prices, pulling from per-ticker CSV cache when possible."""
    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=years)
    freshness_cutoff = today - pd.Timedelta(days=CACHE_STALE_DAYS)

    print(f"Loading prices for {tickers} ({years}y window)...")
    series_by_ticker: dict[str, pd.Series] = {}
    dropped: list[str] = []

    for ticker in tickers:
        cached = cache.load_prices(ticker)
        meta = cache.load_meta(ticker)
        has_cache = cached is not None and not cached.empty

        # Cache "covers the start" when we previously asked yfinance for at least
        # this far back. Whatever it returned is all that exists — comparing
        # against cached.index.min() would falsely miss when start lands on a
        # weekend or before the asset's IPO.
        prior_start_iso = meta.get("earliest_requested_start")
        prior_start = pd.Timestamp(prior_start_iso) if prior_start_iso else None
        covers_start = has_cache and prior_start is not None and prior_start <= start
        fresh = has_cache and cached.index.max() >= freshness_cutoff

        if has_cache and covers_start and fresh:
            print(f"  {ticker}: cache hit  ({len(cached)} rows, through {cached.index.max().date()})")
            series_by_ticker[ticker] = cached["close"]
            continue

        if has_cache and covers_start:
            # Cache covers the window but the tail is stale — incremental update
            fetch_start = cached.index.max() + pd.Timedelta(days=1)
            print(f"  {ticker}: updating   ({fetch_start.date()} → {today.date()})...")
            time.sleep(FETCH_DELAY_SECONDS)
            new_close = fetch.yf_close(ticker, fetch_start, today)
            if not new_close.empty:
                merged = pd.concat([cached["close"], new_close]).sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
                cache.save_prices(ticker, merged, requested_start=start)
                series_by_ticker[ticker] = merged
            else:
                print(f"    ! update failed; using cached data through {cached.index.max().date()}")
                series_by_ticker[ticker] = cached["close"]
            continue

        # No cache, or cache doesn't reach back to requested start — full fetch
        print(f"  {ticker}: full fetch ({start.date()} → {today.date()})...")
        time.sleep(FETCH_DELAY_SECONDS)
        fetched = fetch.yf_close(ticker, start, today)
        if not fetched.empty:
            cache.save_prices(ticker, fetched, requested_start=start)
            series_by_ticker[ticker] = fetched
        elif has_cache:
            print(f"    ! fetch failed; using partial cache "
                  f"({cached.index.min().date()} → {cached.index.max().date()})")
            series_by_ticker[ticker] = cached["close"]
        else:
            print(f"    ! no data and no cache for {ticker} — dropping")
            dropped.append(ticker)

    # Drop anything whose data is obviously corrupt (huge jumps, zero prices).
    for ticker in list(series_by_ticker):
        if not fetch.looks_clean(series_by_ticker[ticker], ticker):
            del series_by_ticker[ticker]
            dropped.append(ticker)

    if dropped:
        print(f"\nWarning: dropped {len(dropped)} ticker(s): {dropped}")
        if not series_by_ticker:
            raise RuntimeError(
                "No usable price data for any ticker. yfinance may be rate-limiting "
                "your IP, or all requested symbols are invalid — try again later or "
                "fix the ticker list."
            )

    prices = pd.DataFrame(series_by_ticker).loc[start:]
    prices = prices.dropna(how="all").ffill().dropna()
    if len(prices) < 100:
        raise RuntimeError(
            f"Only {len(prices)} rows of data — increase --years or check tickers."
        )
    kept = list(prices.columns)
    print(f"Got {len(prices)} trading days of clean data across {len(kept)} tickers: {kept}\n")
    return prices


def fetch_fundamentals(tickers: list[str]) -> dict[str, dict]:
    """Return {ticker: {field: value, ...}}, using cached values up to a day old."""
    today = pd.Timestamp.today().normalize()
    out: dict[str, dict] = {}
    print(f"Loading fundamentals for {tickers}...")

    for ticker in tickers:
        cached = cache.load_fundamentals(ticker)
        if cached and "_fetched" in cached:
            age = today - pd.Timestamp(cached["_fetched"]).normalize()
            if age <= pd.Timedelta(days=FUND_STALE_DAYS):
                print(f"  {ticker}: cache hit  (fetched {cached['_fetched'][:10]})")
                out[ticker] = cached
                continue

        print(f"  {ticker}: refreshing...")
        time.sleep(FETCH_DELAY_SECONDS)
        data = fetch.yf_fundamentals(ticker)
        if data:
            data["_fetched"] = today.isoformat()
            cache.save_fundamentals(ticker, data)
            out[ticker] = data
        elif cached:
            print(f"    ! refresh failed; using stale cache")
            out[ticker] = cached
        else:
            out[ticker] = {}

    return out
