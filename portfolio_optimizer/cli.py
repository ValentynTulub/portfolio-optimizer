"""Command-line entry point.

Usage:
    python optimize.py
    python optimize.py --objective sortino --years 10
    python optimize.py --tickers VGT,VOO,BRK-B,AMZN,KO,GLD --years 15
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from .data import fetch_fundamentals, fetch_prices
from .metrics import portfolio_stats
from .optimization import optimize_weights
from .reporting import (
    correlation_matrix,
    fundamentals_table,
    per_asset_table,
    print_stats,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Portfolio Optimizer — Sortino-focused, with Sharpe and Calmar for comparison."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="VGT,VOO,BRK-B,AMZN,KO,GLD",
        help="Comma-separated tickers (yfinance format, e.g. BRK-B not BRK.B)",
    )
    parser.add_argument("--years", type=int, default=10, help="Years of history")
    parser.add_argument(
        "--objective",
        choices=["sortino", "sharpe", "calmar"],
        default="sortino",
        help="Optimization objective",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.50,
        help="Max weight per asset (caps concentration; 0.5 = 50%%)",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Min weight per asset (e.g. 0.05 to force a 5%% floor)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    tickers = [t.strip() for t in args.tickers.split(",")]
    prices = fetch_prices(tickers, args.years)
    tickers = list(prices.columns)  # narrow to whatever actually came back
    daily_returns = prices.pct_change().dropna()

    per_asset_table(daily_returns)
    correlation_matrix(daily_returns)
    fundamentals_table(fetch_fundamentals(tickers))

    n = len(tickers)
    eq_weights = np.ones(n) / n
    eq_stats = portfolio_stats(eq_weights, daily_returns)
    print_stats("Equal-weight baseline", eq_stats, tickers, eq_weights)

    opt_weights = optimize_weights(
        daily_returns,
        objective=args.objective,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )
    opt_stats = portfolio_stats(opt_weights, daily_returns)
    print_stats(
        f"Optimized for {args.objective.upper()} "
        f"(weights capped at {args.max_weight * 100:.0f}%)",
        opt_stats,
        tickers,
        opt_weights,
    )

    print("\n" + "=" * 60)
    print("  NOTES")
    print("=" * 60)
    print(
        "• Past performance does not predict future returns. The optimizer\n"
        "  fits the historical window — it will overweight whatever happened\n"
        "  to win in that period. Treat output as a starting point, not gospel.\n"
        "• Cap max-weight (e.g. 0.30–0.40) to prevent the optimizer from\n"
        "  dumping everything into one winner.\n"
        "• Try multiple windows (--years 5, 10, 15) to see how stable the\n"
        "  weights are. Stable weights = more trustworthy signal.\n"
        "• Risk-free rate is hardcoded at 4%. Adjust at top of file if needed."
    )
