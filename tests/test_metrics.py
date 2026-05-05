"""Tests for risk/return calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.config import RISK_FREE_RATE, TRADING_DAYS
from portfolio_optimizer.metrics import portfolio_stats


def _returns(values: list[float], cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or ["A"]
    arr = np.array(values).reshape(-1, len(cols))
    idx = pd.date_range("2024-01-01", periods=len(arr), freq="B")
    return pd.DataFrame(arr, index=idx, columns=cols)


def test_constant_positive_return_annualizes_correctly():
    daily = 0.001  # 0.1% per day
    rets = _returns([daily] * 500)
    s = portfolio_stats(np.array([1.0]), rets)

    assert s.annual_return == pytest.approx(daily * TRADING_DAYS)
    assert s.annual_vol == pytest.approx(0.0, abs=1e-9)
    # No negative days → downside_vol falls back to ~0 sentinel; sortino skipped
    assert s.max_drawdown == pytest.approx(0.0, abs=1e-12)
    assert s.calmar == 0  # max_dd not negative → calmar guard returns 0


def test_sharpe_uses_excess_return_over_risk_free():
    rng = np.random.default_rng(0)
    daily = rng.normal(0.001, 0.01, size=2000)
    rets = _returns(list(daily))
    s = portfolio_stats(np.array([1.0]), rets)

    expected = (s.annual_return - RISK_FREE_RATE) / s.annual_vol
    assert s.sharpe == pytest.approx(expected)


def test_sortino_uses_only_downside_deviation():
    """Sortino penalises only sub-zero days, so it should be >= Sharpe for the same series."""
    rng = np.random.default_rng(1)
    daily = rng.normal(0.001, 0.01, size=2000)
    rets = _returns(list(daily))
    s = portfolio_stats(np.array([1.0]), rets)

    # Downside vol is computed from a strict subset of days; can be ≥ or ≤ full vol
    # in degenerate cases, but for normally-distributed data with positive mean,
    # downside vol < total vol → sortino > sharpe.
    assert s.downside_vol < s.annual_vol
    assert s.sortino > s.sharpe


def test_max_drawdown_on_known_path():
    """Construct a path: +10%, -50%, +20% → peak at 1.10, trough at 0.55, MDD = -50%."""
    rets = _returns([0.10, -0.50, 0.20])
    s = portfolio_stats(np.array([1.0]), rets)
    assert s.max_drawdown == pytest.approx(-0.50)


def test_calmar_is_return_over_abs_drawdown():
    rng = np.random.default_rng(2)
    daily = rng.normal(0.0005, 0.01, size=1000)
    rets = _returns(list(daily))
    s = portfolio_stats(np.array([1.0]), rets)

    assert s.calmar == pytest.approx(s.annual_return / abs(s.max_drawdown))


def test_weights_combine_linearly():
    """Equal-weight portfolio of two identical streams equals one stream's stats."""
    rng = np.random.default_rng(3)
    daily = rng.normal(0.0005, 0.01, size=1000)
    rets = _returns(list(daily) + list(daily), cols=["A", "B"])  # not what we want
    # Actually need each row to be [a, a], use a different builder:
    arr = np.column_stack([daily, daily])
    rets = pd.DataFrame(arr, columns=["A", "B"],
                        index=pd.date_range("2024-01-01", periods=1000, freq="B"))

    solo = portfolio_stats(np.array([1.0, 0.0]), rets)
    half = portfolio_stats(np.array([0.5, 0.5]), rets)
    assert solo.annual_return == pytest.approx(half.annual_return)
    assert solo.annual_vol == pytest.approx(half.annual_vol)
