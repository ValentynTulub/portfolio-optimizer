"""Filesystem I/O for cached price series, metadata, and fundamentals."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR


def _safe(ticker: str) -> str:
    return ticker.replace("/", "_")


def price_path(ticker: str) -> Path:
    return CACHE_DIR / f"{_safe(ticker)}.csv"


def meta_path(ticker: str) -> Path:
    return CACHE_DIR / f"{_safe(ticker)}.meta.json"


def fund_path(ticker: str) -> Path:
    return CACHE_DIR / f"{_safe(ticker)}.fund.json"


def load_prices(ticker: str) -> pd.DataFrame | None:
    p = price_path(ticker)
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col="date", parse_dates=["date"])
    df.index = pd.DatetimeIndex(df.index)
    return df  # columns: ["close"]


def load_meta(ticker: str) -> dict:
    p = meta_path(ticker)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_prices(ticker: str, series: pd.Series, requested_start: pd.Timestamp):
    """Persist prices plus a metadata sidecar tracking the earliest start ever requested.

    `requested_start` is the user-asked start date — not the first row of data.
    Storing it lets us recognize "we already tried fetching this far back and that's
    all yfinance has" (e.g. pre-IPO, ETF inception, weekend-rounded boundaries),
    so the next run with the same window can use the cache instead of re-fetching.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    series.to_frame("close").to_csv(price_path(ticker), index_label="date")

    meta = load_meta(ticker)
    earliest = requested_start.date().isoformat()
    prior = meta.get("earliest_requested_start")
    if prior is None or prior > earliest:
        meta["earliest_requested_start"] = earliest
    meta_path(ticker).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_fundamentals(ticker: str) -> dict | None:
    p = fund_path(ticker)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_fundamentals(ticker: str, data: dict):
    CACHE_DIR.mkdir(exist_ok=True)
    fund_path(ticker).write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )
