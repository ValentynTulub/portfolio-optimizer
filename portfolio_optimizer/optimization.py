"""Weight optimization for a chosen risk-adjusted objective."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import portfolio_stats


def optimize_weights(
    daily_returns: pd.DataFrame,
    objective: str = "sortino",
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Find weights maximizing the chosen objective. Long-only, fully invested."""
    n = daily_returns.shape[1]

    # Sum-to-1 with per-asset bounds [min_weight, max_weight] is only feasible
    # when n*max_weight >= 1 and n*min_weight <= 1. Catch this up front, otherwise
    # SLSQP just fails silently from every starting point.
    if n * max_weight < 1.0 - 1e-9:
        raise RuntimeError(
            f"Infeasible: {n} tickers × max_weight={max_weight} = "
            f"{n * max_weight:.2f} < 1.0. Raise --max-weight to at least "
            f"{1.0 / n:.3f}, or add more tickers."
        )
    if n * min_weight > 1.0 + 1e-9:
        raise RuntimeError(
            f"Infeasible: {n} tickers × min_weight={min_weight} = "
            f"{n * min_weight:.2f} > 1.0. Lower --min-weight to at most "
            f"{1.0 / n:.3f}, or remove tickers."
        )

    def neg_objective(w: np.ndarray) -> float:
        stats = portfolio_stats(w, daily_returns)
        return -getattr(stats, objective)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((min_weight, max_weight) for _ in range(n))

    # Try multiple starting points; non-convex objectives like Sortino can have
    # local optima, so we sample a few random simplex points and keep the best.
    best_x, best_val = None, np.inf
    rng = np.random.default_rng(42)
    starts = [np.ones(n) / n]  # equal weight as anchor
    for _ in range(20):
        r = rng.random(n)
        starts.append(r / r.sum())

    for x0 in starts:
        res = minimize(
            neg_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if res.success and res.fun < best_val:
            best_val, best_x = res.fun, res.x

    if best_x is None:
        raise RuntimeError("Optimization failed for all starting points.")
    return best_x
