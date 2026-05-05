"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tmp_cache(tmp_path: Path, monkeypatch) -> Path:
    """Redirect every cache reference to a per-test temp dir.

    Both `cache` and `config` modules hold references to CACHE_DIR; patch each
    so functions in any path resolve to the same isolated directory.
    """
    from portfolio_optimizer import cache, config

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _no_throttle(monkeypatch):
    """Zero out the inter-fetch sleep across the entire test suite."""
    from portfolio_optimizer import data

    monkeypatch.setattr(data, "FETCH_DELAY_SECONDS", 0)


@pytest.fixture
def fake_series():
    """Smooth daily series indexed by trading days. Caller picks length and end-date."""

    def _make(end: pd.Timestamp, periods: int, start_price: float = 100.0,
              end_price: float = 200.0) -> pd.Series:
        dates = pd.bdate_range(end=end, periods=periods)
        # Linear ramp keeps `looks_clean` happy (no >200% jumps).
        import numpy as np
        values = np.linspace(start_price, end_price, len(dates))
        return pd.Series(values, index=dates, name="close")

    return _make
