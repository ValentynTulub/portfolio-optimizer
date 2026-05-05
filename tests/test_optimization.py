"""Tests for the SLSQP weight optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.optimization import optimize_weights


def _two_asset_returns(seed: int = 0, n: int = 1000) -> pd.DataFrame:
    """A is a clearly better risk-adjusted asset than B."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.001, 0.008, size=n)   # higher mean, lower vol
    b = rng.normal(0.0002, 0.020, size=n)  # lower mean, higher vol
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_infeasible_when_max_weight_too_low():
    rets = _two_asset_returns()
    # 2 assets * 0.4 = 0.8 < 1.0 → cannot satisfy sum-to-1
    with pytest.raises(RuntimeError, match="Infeasible"):
        optimize_weights(rets, max_weight=0.4)


def test_infeasible_when_min_weight_too_high():
    rets = _two_asset_returns()
    # 2 assets * 0.6 = 1.2 > 1.0 → cannot satisfy sum-to-1
    with pytest.raises(RuntimeError, match="Infeasible"):
        optimize_weights(rets, min_weight=0.6, max_weight=1.0)


def test_weights_sum_to_one():
    rets = _two_asset_returns()
    w = optimize_weights(rets, objective="sortino")
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_weights_respect_bounds():
    rets = _two_asset_returns()
    w = optimize_weights(rets, min_weight=0.1, max_weight=0.7)
    assert (w >= 0.1 - 1e-6).all()
    assert (w <= 0.7 + 1e-6).all()


def test_optimizer_prefers_better_risk_adjusted_asset():
    """With A having a clearly better Sortino than B, the optimizer should weight A higher."""
    rets = _two_asset_returns()
    w = optimize_weights(rets, objective="sortino")
    a_idx = list(rets.columns).index("A")
    b_idx = list(rets.columns).index("B")
    assert w[a_idx] > w[b_idx]


def test_optimizer_concentrates_under_loose_cap():
    """Without a tight cap, the optimizer should push the dominant asset toward its bound."""
    rets = _two_asset_returns()
    w = optimize_weights(rets, objective="sortino", max_weight=1.0)
    a_idx = list(rets.columns).index("A")
    assert w[a_idx] > 0.9  # essentially all in A


def test_max_weight_cap_actually_caps():
    rets = _two_asset_returns()
    w = optimize_weights(rets, objective="sortino", max_weight=0.6)
    a_idx = list(rets.columns).index("A")
    assert w[a_idx] == pytest.approx(0.6, abs=1e-4)
