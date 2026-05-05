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

from .backtest import split_returns
from .data import fetch_fundamentals, fetch_prices
from .metrics import portfolio_stats
from .optimization import optimize_weights
from .reporting import (
    correlation_matrix,
    fundamentals_table,
    per_asset_table,
    print_backtest_comparison,
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
    parser.add_argument(
        "--benchmark-n-last-years",
        type=int,
        default=0,
        metavar="N",
        help="Walk-forward backtest: fit on (years - N), then evaluate the fitted "
             "weights on the held-out last N years. 0 = disabled (default).",
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
    fundamentals = fetch_fundamentals(tickers)
    fundamentals_table(fundamentals)

    n = len(tickers)
    eq_weights = np.ones(n) / n

    if args.benchmark_n_last_years > 0:
        if args.benchmark_n_last_years >= args.years:
            raise SystemExit(
                f"--benchmark-n-last-years ({args.benchmark_n_last_years}) must be "
                f"strictly less than --years ({args.years})."
            )
        train_returns, test_returns = split_returns(
            daily_returns, args.benchmark_n_last_years
        )
        if len(train_returns) < 100 or len(test_returns) < 100:
            raise SystemExit(
                f"Backtest split too small: train={len(train_returns)} rows, "
                f"test={len(test_returns)} rows. Need ≥100 each."
            )

        eq_train = portfolio_stats(eq_weights, train_returns)
        print_stats(
            f"Equal-weight baseline (fit window, {args.years - args.benchmark_n_last_years}y)",
            eq_train, tickers, eq_weights,
        )

        opt_weights = optimize_weights(
            train_returns,
            objective=args.objective,
            min_weight=args.min_weight,
            max_weight=args.max_weight,
        )
        opt_in_sample = portfolio_stats(opt_weights, train_returns)
        opt_out_of_sample = portfolio_stats(opt_weights, test_returns)
        eq_out_of_sample = portfolio_stats(eq_weights, test_returns)

        print_stats(
            f"Optimized for {args.objective.upper()} "
            f"(fit window only, weights capped at {args.max_weight * 100:.0f}%)",
            opt_in_sample, tickers, opt_weights,
            fundamentals=fundamentals,
        )

        print_backtest_comparison(
            opt_in_sample, opt_out_of_sample,
            fit_window=(train_returns.index.min(), train_returns.index.max()),
            oos_window=(test_returns.index.min(), test_returns.index.max()),
            fit_rows=len(train_returns),
            oos_rows=len(test_returns),
        )

        # Equal-weight on the same OOS slice — the control group.
        # If the optimizer's OOS Sortino is below this, the fit didn't generalize.
        print_stats(
            f"Equal-weight on out-of-sample window ({args.benchmark_n_last_years}y, control)",
            eq_out_of_sample, tickers, eq_weights,
        )
    else:
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
            opt_stats, tickers, opt_weights,
            fundamentals=fundamentals,
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
