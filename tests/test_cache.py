"""Tests for filesystem I/O of price/meta/fundamentals."""

from __future__ import annotations

import json

import pandas as pd

from portfolio_optimizer import cache


def test_load_prices_returns_none_when_missing(tmp_cache):
    assert cache.load_prices("NOPE") is None


def test_load_meta_returns_empty_when_missing(tmp_cache):
    assert cache.load_meta("NOPE") == {}


def test_load_meta_returns_empty_on_corrupt_json(tmp_cache):
    cache.meta_path("AAPL").write_text("{not json", encoding="utf-8")
    assert cache.load_meta("AAPL") == {}


def test_save_load_prices_roundtrip(tmp_cache, fake_series):
    # 2024-05-31 is a Friday, so bdate_range produces exactly `periods` rows
    series = fake_series(end=pd.Timestamp("2024-05-31"), periods=10)
    requested_start = pd.Timestamp("2014-06-01")

    cache.save_prices("AAPL", series, requested_start=requested_start)

    loaded = cache.load_prices("AAPL")
    assert loaded is not None
    assert list(loaded.columns) == ["close"]
    assert len(loaded) == len(series)
    assert loaded["close"].iloc[0] == series.iloc[0]
    assert loaded.index.max() == series.index.max()


def test_save_prices_writes_meta(tmp_cache, fake_series):
    series = fake_series(end=pd.Timestamp("2024-05-31"), periods=10)
    cache.save_prices("AAPL", series, requested_start=pd.Timestamp("2014-06-01"))

    meta = cache.load_meta("AAPL")
    assert meta["earliest_requested_start"] == "2014-06-01"


def test_meta_keeps_smallest_start_across_writes(tmp_cache, fake_series):
    """Asking for an earlier start lowers the recorded earliest_requested_start.
    Asking for a later one must NOT raise it — the cache covers more, not less."""
    series = fake_series(end=pd.Timestamp("2024-06-01"), periods=10)

    cache.save_prices("AAPL", series, requested_start=pd.Timestamp("2014-06-01"))
    assert cache.load_meta("AAPL")["earliest_requested_start"] == "2014-06-01"

    # Later start — must not regress the earliest tracker
    cache.save_prices("AAPL", series, requested_start=pd.Timestamp("2020-01-01"))
    assert cache.load_meta("AAPL")["earliest_requested_start"] == "2014-06-01"

    # Earlier start — should lower it
    cache.save_prices("AAPL", series, requested_start=pd.Timestamp("2001-05-05"))
    assert cache.load_meta("AAPL")["earliest_requested_start"] == "2001-05-05"


def test_save_load_fundamentals_roundtrip(tmp_cache):
    payload = {"trailingPE": 27.5, "_fetched": "2024-06-01T00:00:00"}
    cache.save_fundamentals("AAPL", payload)

    loaded = cache.load_fundamentals("AAPL")
    assert loaded == payload


def test_load_fundamentals_returns_none_on_corrupt_json(tmp_cache):
    cache.fund_path("AAPL").write_text("{nope", encoding="utf-8")
    assert cache.load_fundamentals("AAPL") is None


def test_path_helpers_sanitize_slashes(tmp_cache):
    """yfinance allows tickers with slashes (rare); they shouldn't escape the cache dir."""
    p = cache.price_path("FOO/BAR")
    assert "/" not in p.name
    assert p.parent == tmp_cache
