"""Console-friendly formatting of portfolio results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import PortfolioStats, portfolio_stats


def print_stats(label: str, stats: PortfolioStats, tickers: list[str], weights: np.ndarray):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print("Weights:")
    for t, w in sorted(zip(tickers, weights), key=lambda x: -x[1]):
        bar = "█" * int(w * 40)
        print(f"  {t:8s} {w * 100:6.2f}%  {bar}")
    print()
    print(f"  Annual return    : {stats.annual_return * 100:6.2f}%")
    print(f"  Annual volatility: {stats.annual_vol * 100:6.2f}%")
    print(f"  Downside vol     : {stats.downside_vol * 100:6.2f}%")
    print(f"  Sharpe ratio     : {stats.sharpe:6.3f}")
    print(f"  Sortino ratio    : {stats.sortino:6.3f}  ← downside-only")
    print(f"  Max drawdown     : {stats.max_drawdown * 100:6.2f}%")
    print(f"  Calmar ratio     : {stats.calmar:6.3f}  ← return / max DD")


def per_asset_table(daily_returns: pd.DataFrame):
    print(f"\n{'=' * 60}")
    print("  Individual asset stats")
    print(f"{'=' * 60}")
    print(f"  {'Ticker':<8} {'Return':>8} {'Vol':>8} {'Sharpe':>8} {'Sortino':>9} {'MaxDD':>9}")
    for t in daily_returns.columns:
        w = np.zeros(daily_returns.shape[1])
        w[list(daily_returns.columns).index(t)] = 1.0
        s = portfolio_stats(w, daily_returns)
        print(
            f"  {t:<8} "
            f"{s.annual_return * 100:7.2f}% "
            f"{s.annual_vol * 100:7.2f}% "
            f"{s.sharpe:8.3f} "
            f"{s.sortino:9.3f} "
            f"{s.max_drawdown * 100:8.2f}%"
        )


def correlation_matrix(daily_returns: pd.DataFrame):
    print(f"\n{'=' * 60}")
    print("  Correlation matrix")
    print(f"{'=' * 60}")
    corr = daily_returns.corr()
    print(corr.round(2).to_string())


def fundamentals_table(fundamentals: dict[str, dict]):
    print(f"\n{'=' * 60}")
    print("  Valuation snapshot")
    print(f"{'=' * 60}")
    print(f"  {'Ticker':<8} {'P/E':>8} {'Fwd P/E':>8} {'PEG':>6} {'P/B':>6} {'Div%':>6}  notes")

    pes = [f["trailingPE"] for f in fundamentals.values()
           if isinstance(f.get("trailingPE"), (int, float)) and f["trailingPE"] > 0]
    pe_median = sorted(pes)[len(pes) // 2] if pes else None

    def _fmt(v, width: int, spec: str) -> str:
        return format(v, spec) if isinstance(v, (int, float)) else " " * (width - 3) + "n/a"

    for ticker, f in fundamentals.items():
        pe = f.get("trailingPE")
        fpe = f.get("forwardPE")
        peg = f.get("pegRatio")
        pb = f.get("priceToBook")
        dy = f.get("dividendYield")

        # yfinance flips between decimal (0.015) and percent (1.5) for dividendYield
        # depending on version; normalize to percent.
        dy_pct = (dy * 100 if isinstance(dy, (int, float)) and dy < 1 else dy)

        notes = []
        if (isinstance(pe, (int, float)) and pe > 0
                and pe_median is not None and pe < pe_median):
            notes.append("below basket median P/E")
        if (isinstance(fpe, (int, float)) and isinstance(pe, (int, float))
                and fpe > 0 and pe > 0 and fpe < pe * 0.95):
            notes.append("earnings expected to grow")
        if isinstance(peg, (int, float)) and 0 < peg < 1:
            notes.append("PEG<1 (cheap vs growth)")

        print(
            f"  {ticker:<8} "
            f"{_fmt(pe, 8, '8.1f')} "
            f"{_fmt(fpe, 8, '8.1f')} "
            f"{_fmt(peg, 6, '6.2f')} "
            f"{_fmt(pb, 6, '6.2f')} "
            f"{_fmt(dy_pct, 6, '5.2f') + '%' if isinstance(dy_pct, (int, float)) else '   n/a'}"
            f"  {', '.join(notes)}"
        )

    print(
        "\n  Notes: P/E ratios are current snapshots, not historical percentiles —\n"
        "  a low number relative to peers/history *suggests* a discount but doesn't\n"
        "  prove one. Commodity ETFs (e.g. GLD) have no earnings, so P/E is n/a."
    )
