"""Console-friendly formatting of portfolio results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import PortfolioStats, portfolio_stats


def _num_or_na(v, spec: str, width: int) -> str:
    """Format a numeric metric to fixed width, or right-pad 'n/a' to the same width."""
    return format(v, spec) if isinstance(v, (int, float)) else "n/a".rjust(width)


def _format_fundamentals_inline(f: dict) -> str:
    """One-line snapshot of P/E, PEG, P/B, Div% for the weights view.

    Widths chosen to handle the awkward edge cases that show up in real data:
    negative P/B (Liberty share classes carry it), 3-digit P/E, and so on —
    so columns align even when those rows appear.
    """
    pe = f.get("trailingPE")
    peg = f.get("pegRatio")
    pb = f.get("priceToBook")
    dy = f.get("dividendYield")  # yfinance ≥ ~0.2.30 returns percent already

    div_str = f"{dy:5.2f}%" if isinstance(dy, (int, float)) else "n/a".rjust(6)

    return (
        f"P/E {_num_or_na(pe, '6.1f', 6)}    "
        f"PEG {_num_or_na(peg, '5.2f', 5)}    "
        f"P/B {_num_or_na(pb, '7.2f', 7)}    "
        f"Div {div_str}"
    )


def _make_bar(weight: float, bar_width: int) -> str:
    """Filled-block bar (1 char per 2.5%) padded to bar_width with ░ for unfilled.

    Truly-zero weights render as whitespace so empty rows aren't visually noisy.
    """
    if weight <= 1e-4:
        return " " * bar_width
    fill = min(int(weight * 40), bar_width)
    return "█" * fill + "░" * (bar_width - fill)


def print_stats(
    label: str,
    stats: PortfolioStats,
    tickers: list[str],
    weights: np.ndarray,
    fundamentals: dict[str, dict] | None = None,
):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print("Weights:")

    # Bar width = widest single bar in this view, with a minimum for breathing
    # room when caps keep all positions small (e.g. max_weight=0.10 → 4-char bars).
    max_fill = max((int(w * 40) for w in weights), default=0)
    bar_width = max(max_fill, 10)

    for t, w in sorted(zip(tickers, weights), key=lambda x: -x[1]):
        bar = _make_bar(w, bar_width)
        line = f"  {t:8s} {w * 100:6.2f}%   {bar}"
        if fundamentals is not None and w > 1e-4:
            f = fundamentals.get(t) or {}
            line += "    " + _format_fundamentals_inline(f)
        print(line)
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
        dy = f.get("dividendYield")  # current yfinance returns percent

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
            f"{_fmt(dy, 6, '5.2f') + '%' if isinstance(dy, (int, float)) else '   n/a'}"
            f"  {', '.join(notes)}"
        )

    print(
        "\n  Notes: P/E ratios are current snapshots, not historical percentiles —\n"
        "  a low number relative to peers/history *suggests* a discount but doesn't\n"
        "  prove one. Commodity ETFs (e.g. GLD) have no earnings, so P/E is n/a."
    )
