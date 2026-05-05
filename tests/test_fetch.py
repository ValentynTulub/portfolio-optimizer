"""Tests for the data-quality validator. yfinance wrappers are skipped (network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimizer import fetch


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="close")


def test_looks_clean_accepts_normal_series():
    rng = np.random.default_rng(0)
    daily = 1 + rng.normal(0.0005, 0.01, size=500)
    prices = 100 * np.cumprod(daily)
    assert fetch.looks_clean(_series(list(prices)), "TEST") is True


def test_looks_clean_rejects_empty():
    assert fetch.looks_clean(pd.Series(dtype=float, name="close"), "TEST") is False


def test_looks_clean_rejects_single_point():
    assert fetch.looks_clean(_series([100.0]), "TEST") is False


def test_looks_clean_rejects_zero_price():
    assert fetch.looks_clean(_series([100.0, 0.0, 100.0]), "TEST") is False


def test_looks_clean_rejects_negative_price():
    assert fetch.looks_clean(_series([100.0, -1.0, 100.0]), "TEST") is False


def test_looks_clean_rejects_extreme_jump():
    """A >200% one-day move signals spliced/corrupt yfinance data."""
    assert fetch.looks_clean(_series([100.0, 100.0, 500.0, 500.0]), "TEST") is False
