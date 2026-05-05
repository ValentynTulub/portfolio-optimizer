"""Risk/return metrics for a weight vector against a daily-return matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RISK_FREE_RATE, TRADING_DAYS


@dataclass
class PortfolioStats:
    annual_return: float
    annual_vol: float
    downside_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float


def portfolio_stats(weights: np.ndarray, daily_returns: pd.DataFrame) -> PortfolioStats:
    """Compute annualized metrics for a given weight vector."""
    port_daily = daily_returns @ weights

    annual_return = port_daily.mean() * TRADING_DAYS
    annual_vol = port_daily.std(ddof=1) * np.sqrt(TRADING_DAYS)

    # Downside deviation: only count returns below zero (target = 0)
    downside = port_daily[port_daily < 0]
    downside_vol = (
        downside.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(downside) > 1 else 1e-9
    )

    sharpe = (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else 0
    sortino = (annual_return - RISK_FREE_RATE) / downside_vol if downside_vol > 0 else 0

    # Max drawdown from cumulative returns
    cum = (1 + port_daily).cumprod()
    running_peak = cum.cummax()
    drawdown = (cum - running_peak) / running_peak
    max_dd = drawdown.min()

    calmar = annual_return / abs(max_dd) if max_dd < 0 else 0

    return PortfolioStats(
        annual_return=annual_return,
        annual_vol=annual_vol,
        downside_vol=downside_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
    )
