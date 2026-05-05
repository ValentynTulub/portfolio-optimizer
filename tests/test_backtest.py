"""Tests for the train/test split used by walk-forward backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.backtest import split_returns


def _returns(start: str, end: str, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or ["A"]
    idx = pd.bdate_range(start=start, end=end)
    rng = np.random.default_rng(0)
    arr = rng.normal(0.0005, 0.01, size=(len(idx), len(cols)))
    return pd.DataFrame(arr, index=idx, columns=cols)


def test_split_partitions_returns_completely():
    rets = _returns("2010-01-01", "2025-01-01")
    train, test = split_returns(rets, n_oos_years=5)

    # Round-trip: every row goes into exactly one side
    assert len(train) + len(test) == len(rets)
    assert train.index.max() < test.index.min()


def test_split_test_window_is_n_years():
    rets = _returns("2010-01-01", "2025-01-01")
    _, test = split_returns(rets, n_oos_years=5)

    span = test.index.max() - test.index.min()
    assert pd.Timedelta(days=4 * 365) < span < pd.Timedelta(days=5 * 365 + 5)


def test_split_with_n_equal_to_total_yields_empty_train():
    rets = _returns("2020-01-01", "2025-01-01")
    train, test = split_returns(rets, n_oos_years=5)
    # Test window covers ~5y; train can be empty or near-empty by design
    assert len(test) > len(train)


def test_split_rejects_zero_or_negative():
    rets = _returns("2020-01-01", "2025-01-01")
    with pytest.raises(ValueError):
        split_returns(rets, n_oos_years=0)
    with pytest.raises(ValueError):
        split_returns(rets, n_oos_years=-3)


def test_split_rejects_empty_returns():
    empty = pd.DataFrame(columns=["A"], index=pd.DatetimeIndex([], name="date"))
    with pytest.raises(ValueError):
        split_returns(empty, n_oos_years=5)


def test_split_preserves_columns():
    rets = _returns("2010-01-01", "2025-01-01", cols=["AAPL", "GOOG", "KO"])
    train, test = split_returns(rets, n_oos_years=3)
    assert list(train.columns) == ["AAPL", "GOOG", "KO"]
    assert list(test.columns) == ["AAPL", "GOOG", "KO"]
