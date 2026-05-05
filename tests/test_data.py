"""Tests for the cache+fetch orchestration in fetch_prices.

The branch matrix:
  has_cache | covers_start | fresh | action
  ----------+--------------+-------+--------------------
    no      |      —       |   —   | full fetch
    yes     |     no       |   —   | full fetch (extends back)
    yes     |     yes      |  no   | incremental update (tail)
    yes     |     yes      |  yes  | cache hit (no network)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer import cache, data


@pytest.fixture
def no_network(monkeypatch):
    """Replace yfinance wrappers with a recorder that returns empty by default.

    Tests that expect a fetch override .return_value; tests that expect a cache
    hit assert .call_args_list stays empty.
    """
    calls: list[dict] = []

    def fake_yf_close(ticker, start, end):
        calls.append({"ticker": ticker, "start": start, "end": end})
        return fake_yf_close.return_value

    fake_yf_close.return_value = pd.Series(dtype=float, name="close")
    fake_yf_close.calls = calls

    monkeypatch.setattr("portfolio_optimizer.fetch.yf_close", fake_yf_close)
    return fake_yf_close


def _populate_cache(ticker: str, end: pd.Timestamp, periods: int,
                    requested_start: pd.Timestamp) -> pd.Series:
    dates = pd.bdate_range(end=end, periods=periods)
    series = pd.Series(np.linspace(100, 200, len(dates)), index=dates, name="close")
    cache.save_prices(ticker, series, requested_start=requested_start)
    return series


def test_cache_hit_skips_network(tmp_cache, no_network):
    today = pd.Timestamp.today().normalize()
    # Cache covers 11y back, last row = today (fresh), prior_start = 11y ago
    _populate_cache("AAPL", end=today, periods=2500,
                    requested_start=today - pd.DateOffset(years=11))

    prices = data.fetch_prices(["AAPL"], years=10)

    assert no_network.calls == [], "should not call yfinance on a fresh, covering cache"
    assert "AAPL" in prices.columns


def test_stale_cache_triggers_incremental_update(tmp_cache, no_network):
    today = pd.Timestamp.today().normalize()
    cached = _populate_cache("AAPL", end=today - pd.Timedelta(days=10), periods=2500,
                             requested_start=today - pd.DateOffset(years=11))
    last_cached_day = cached.index.max()  # bdate_range may snap from a weekend

    # Return one new bar from the "network"
    no_network.return_value = pd.Series(
        [250.0],
        index=pd.DatetimeIndex([last_cached_day + pd.Timedelta(days=3)]),
        name="close",
    )

    data.fetch_prices(["AAPL"], years=10)

    assert len(no_network.calls) == 1
    call = no_network.calls[0]
    assert call["ticker"] == "AAPL"
    # Incremental fetch should start the day after the last cached row, not from scratch
    assert call["start"] == last_cached_day + pd.Timedelta(days=1)


def test_cache_with_no_meta_triggers_full_fetch(tmp_cache, no_network, fake_series):
    """A pre-existing CSV with no meta sidecar (legacy state) must re-fetch once
    so that meta gets written — otherwise we can never reach a cache hit."""
    today = pd.Timestamp.today().normalize()
    series = fake_series(end=today, periods=2500)
    # Bypass save_prices so meta is NOT written, simulating legacy cache
    series.to_frame("close").to_csv(cache.price_path("AAPL"), index_label="date")

    no_network.return_value = series  # any non-empty series so fetch counts as success
    data.fetch_prices(["AAPL"], years=10)

    assert len(no_network.calls) == 1, "missing meta should force a full fetch"
    # And after the fetch, meta exists with the requested start
    meta = cache.load_meta("AAPL")
    assert "earliest_requested_start" in meta


def test_meta_with_later_start_than_requested_triggers_full_fetch(tmp_cache, no_network):
    """Regression test for the cache-miss bug: a 5y-old meta against a 10y request
    should re-fetch (not falsely hit cache). After fetching, meta moves to 10y."""
    today = pd.Timestamp.today().normalize()
    _populate_cache("AAPL", end=today, periods=1500,
                    requested_start=today - pd.DateOffset(years=5))

    no_network.return_value = pd.Series(
        np.linspace(50, 200, 2500),
        index=pd.bdate_range(end=today, periods=2500),
        name="close",
    )
    data.fetch_prices(["AAPL"], years=10)

    assert len(no_network.calls) == 1
    # Meta should now reflect the larger window
    meta = cache.load_meta("AAPL")
    new_start = pd.Timestamp(meta["earliest_requested_start"])
    assert new_start <= today - pd.DateOffset(years=10) + pd.Timedelta(days=1)


def test_weekend_boundary_does_not_force_refetch(tmp_cache, no_network):
    """The bug we fixed: requested start lands on a weekend, cache's first row is
    the next Monday. With meta tracking, this should remain a cache hit."""
    today = pd.Timestamp.today().normalize()
    requested_start = today - pd.DateOffset(years=10)
    # Cache's earliest row is 2 days AFTER requested_start (weekend skipped)
    cache_start = requested_start + pd.Timedelta(days=2)
    dates = pd.bdate_range(start=cache_start, end=today)
    series = pd.Series(np.linspace(100, 200, len(dates)), index=dates, name="close")
    cache.save_prices("AAPL", series, requested_start=requested_start)

    data.fetch_prices(["AAPL"], years=10)

    assert no_network.calls == [], (
        "weekend-rounded cache.min must not trick the predicate into refetching"
    )


def test_pre_ipo_request_does_not_force_refetch(tmp_cache, no_network):
    """Asking for 25y of GLD when GLD only has data from 2004 should still be a
    cache hit on subsequent runs — yfinance has no earlier data to deliver."""
    today = pd.Timestamp.today().normalize()
    # Asset 'inception' is well after the 25y-ago start
    inception = today - pd.DateOffset(years=20)
    requested_start = today - pd.DateOffset(years=25)
    dates = pd.bdate_range(start=inception, end=today)
    series = pd.Series(np.linspace(100, 200, len(dates)), index=dates, name="close")
    cache.save_prices("GLD", series, requested_start=requested_start)

    data.fetch_prices(["GLD"], years=25)

    assert no_network.calls == [], "post-inception data is all that exists; no need to refetch"


def test_no_cache_no_network_drops_ticker(tmp_cache, no_network):
    """If the cache is empty AND yfinance returns nothing, the ticker is dropped.
    With no usable tickers, the function raises rather than returning empty data."""
    # no_network.return_value already empty by default
    with pytest.raises(RuntimeError, match="No usable price data"):
        data.fetch_prices(["NOPE"], years=10)


def test_corrupt_cached_series_dropped_by_looks_clean(tmp_cache, no_network):
    """A cache hit that's clearly corrupt (>200% jumps) should still be dropped
    — the cleanliness check runs after the cache decision."""
    today = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=today, periods=2500)
    bad = np.linspace(100, 200, len(dates))
    bad[1000] = 5000.0  # absurd one-day spike
    series = pd.Series(bad, index=dates, name="close")
    cache.save_prices("BAD", series, requested_start=today - pd.DateOffset(years=11))

    with pytest.raises(RuntimeError, match="No usable price data"):
        data.fetch_prices(["BAD"], years=10)
