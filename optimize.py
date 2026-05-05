"""
Portfolio Optimizer — Sortino-focused, with Sharpe and Calmar for comparison.

Pulls historical adjusted-close data via yfinance, computes risk metrics,
and finds the weight vector that maximizes the chosen objective.

Usage:
    python optimize.py
    python optimize.py --objective sortino --years 10
    python optimize.py --tickers VGT,VOO,BRK-B,AMZN,KO,GLD --years 15
"""

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=FutureWarning)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04  # ~4% annual; adjust to current 10Y treasury if you like

CACHE_DIR = Path(__file__).parent / "ticker_cache"
_CACHE_STALE_DAYS = 4  # tolerate up to 4-day gap (weekends + 1 holiday)
_FUND_STALE_DAYS = 1  # fundamentals barely move intraday; refresh daily

_FUND_FIELDS = (
    "trailingPE",
    "forwardPE",
    "pegRatio",
    "priceToBook",
    "dividendYield",
    "trailingEps",
    "forwardEps",
)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('/', '_')}.csv"


def _meta_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('/', '_')}.meta.json"


def _load_cache(ticker: str) -> pd.DataFrame | None:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col="date", parse_dates=["date"])
    df.index = pd.DatetimeIndex(df.index)
    return df  # columns: ["close"]


def _load_meta(ticker: str) -> dict:
    p = _meta_path(ticker)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(ticker: str, series: pd.Series, requested_start: pd.Timestamp):
    """Persist prices plus a metadata sidecar tracking the earliest start ever requested.

    `requested_start` is the user-asked start date — not the first row of data.
    Storing it lets us recognize "we already tried fetching this far back and that's
    all yfinance has" (e.g. pre-IPO, ETF inception, weekend-rounded boundaries),
    so the next run with the same window can use the cache instead of re-fetching.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    series.to_frame("close").to_csv(_cache_path(ticker), index_label="date")

    meta = _load_meta(ticker)
    prior = meta.get("earliest_requested_start")
    earliest = requested_start.date().isoformat()
    if prior is None or prior > earliest:
        meta["earliest_requested_start"] = earliest
    _meta_path(ticker).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _looks_clean(close: pd.Series, ticker: str) -> bool:
    """Reject series with non-positive prices or absurd single-day jumps.

    yfinance occasionally returns data spliced across reused symbols — e.g. an
    old delisted security and a newer one sharing the same ticker — which
    produces multi-thousand-percent daily moves that wreck the optimizer.
    """
    if close.empty or len(close) < 2:
        print(f"    ! {ticker}: too few points ({len(close)}) — dropping")
        return False
    if (close <= 0).any():
        print(f"    ! {ticker}: contains non-positive prices — dropping")
        return False
    rets = close.pct_change().dropna()
    extreme = rets[rets.abs() > 2.0]  # >200% in one day → almost certainly bad
    if not extreme.empty:
        worst = extreme.iloc[extreme.abs().argmax()]
        when = extreme.index[extreme.abs().argmax()].date()
        print(
            f"    ! {ticker}: {len(extreme)} day(s) with |return|>200% "
            f"(worst {worst:+.0%} on {when}) — likely spliced data, dropping. "
            f"Delete {_cache_path(ticker).name} to retry fetch."
        )
        return False
    return True


def _yf_close(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch adjusted closes for one ticker. Returns empty Series on any failure."""
    try:
        raw = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False,
        )
    except Exception as e:
        print(f"    ! yfinance error for {ticker}: {type(e).__name__}: {e}")
        return pd.Series(dtype=float, name="close")

    if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
        return pd.Series(dtype=float, name="close")

    close = raw["Close"]
    # Newer yfinance returns MultiIndex columns even for a single ticker;
    # raw["Close"] is then a 1-col DataFrame rather than a Series.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.name = "close"
    return close


def _fund_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('/', '_')}.fund.json"


def _load_fund_cache(ticker: str) -> dict | None:
    p = _fund_path(ticker)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_fund_cache(ticker: str, data: dict):
    CACHE_DIR.mkdir(exist_ok=True)
    _fund_path(ticker).write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )


def _yf_fundamentals(ticker: str) -> dict:
    """Pull a small whitelist of valuation fields from yfinance. Empty dict on failure."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        print(f"    ! fundamentals error for {ticker}: {type(e).__name__}: {e}")
        return {}
    return {k: info.get(k) for k in _FUND_FIELDS if info.get(k) is not None}


def fetch_fundamentals(tickers: list[str]) -> dict[str, dict]:
    """Return {ticker: {field: value, ...}}, using cached values up to a day old."""
    today = pd.Timestamp.today().normalize()
    out: dict[str, dict] = {}
    print(f"Loading fundamentals for {tickers}...")

    for ticker in tickers:
        cached = _load_fund_cache(ticker)
        if cached and "_fetched" in cached:
            age = today - pd.Timestamp(cached["_fetched"]).normalize()
            if age <= pd.Timedelta(days=_FUND_STALE_DAYS):
                print(f"  {ticker}: cache hit  (fetched {cached['_fetched'][:10]})")
                out[ticker] = cached
                continue

        print(f"  {ticker}: refreshing...")
        data = _yf_fundamentals(ticker)
        if data:
            data["_fetched"] = today.isoformat()
            _save_fund_cache(ticker, data)
            out[ticker] = data
        elif cached:
            print(f"    ! refresh failed; using stale cache")
            out[ticker] = cached
        else:
            out[ticker] = {}

    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_prices(tickers: list[str], years: int) -> pd.DataFrame:
    """Return adjusted close prices, pulling from per-ticker CSV cache when possible."""
    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=years)
    freshness_cutoff = today - pd.Timedelta(days=_CACHE_STALE_DAYS)

    print(f"Loading prices for {tickers} ({years}y window)...")
    series_by_ticker: dict[str, pd.Series] = {}
    dropped: list[str] = []

    for ticker in tickers:
        cached = _load_cache(ticker)
        meta = _load_meta(ticker)
        has_cache = cached is not None and not cached.empty

        # Cache "covers the start" when we previously asked yfinance for at least
        # this far back. Whatever it returned is all that exists — comparing
        # against cached.index.min() would falsely miss when start lands on a
        # weekend or before the asset's IPO.
        prior_start_iso = meta.get("earliest_requested_start")
        prior_start = pd.Timestamp(prior_start_iso) if prior_start_iso else None
        covers_start = has_cache and prior_start is not None and prior_start <= start
        fresh = has_cache and cached.index.max() >= freshness_cutoff

        if has_cache and covers_start and fresh:
            print(f"  {ticker}: cache hit  ({len(cached)} rows, through {cached.index.max().date()})")
            series_by_ticker[ticker] = cached["close"]
            continue

        if has_cache and covers_start:
            # Cache covers the window but the tail is stale — incremental update
            fetch_start = cached.index.max() + pd.Timedelta(days=1)
            print(f"  {ticker}: updating   ({fetch_start.date()} → {today.date()})...")
            new_close = _yf_close(ticker, fetch_start, today)
            if not new_close.empty:
                merged = pd.concat([cached["close"], new_close]).sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
                _save_cache(ticker, merged, requested_start=start)
                series_by_ticker[ticker] = merged
            else:
                print(f"    ! update failed; using cached data through {cached.index.max().date()}")
                series_by_ticker[ticker] = cached["close"]
            continue

        # No cache, or cache doesn't reach back to requested start — full fetch
        print(f"  {ticker}: full fetch ({start.date()} → {today.date()})...")
        fetched = _yf_close(ticker, start, today)
        if not fetched.empty:
            _save_cache(ticker, fetched, requested_start=start)
            series_by_ticker[ticker] = fetched
        elif has_cache:
            print(f"    ! fetch failed; using partial cache "
                  f"({cached.index.min().date()} → {cached.index.max().date()})")
            series_by_ticker[ticker] = cached["close"]
        else:
            print(f"    ! no data and no cache for {ticker} — dropping")
            dropped.append(ticker)

    # Drop anything whose data is obviously corrupt (huge jumps, zero prices).
    for ticker in list(series_by_ticker):
        if not _looks_clean(series_by_ticker[ticker], ticker):
            del series_by_ticker[ticker]
            dropped.append(ticker)

    if dropped:
        print(f"\nWarning: dropped {len(dropped)} ticker(s): {dropped}")
        if not series_by_ticker:
            raise RuntimeError(
                "No usable price data for any ticker. yfinance may be rate-limiting "
                "your IP, or all requested symbols are invalid — try again later or "
                "fix the ticker list."
            )

    prices = pd.DataFrame(series_by_ticker).loc[start:]
    prices = prices.dropna(how="all").ffill().dropna()
    if len(prices) < 100:
        raise RuntimeError(
            f"Only {len(prices)} rows of data — increase --years or check tickers."
        )
    kept = list(prices.columns)
    print(f"Got {len(prices)} trading days of clean data across {len(kept)} tickers: {kept}\n")
    return prices


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def optimize(
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

    # Constraints: weights sum to 1
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]
    prices = fetch_prices(tickers, args.years)
    tickers = list(prices.columns)  # narrow to whatever actually came back
    daily_returns = prices.pct_change().dropna()

    per_asset_table(daily_returns)
    correlation_matrix(daily_returns)
    fundamentals_table(fetch_fundamentals(tickers))

    # Equal-weight baseline
    n = len(tickers)
    eq_weights = np.ones(n) / n
    eq_stats = portfolio_stats(eq_weights, daily_returns)
    print_stats("Equal-weight baseline", eq_stats, tickers, eq_weights)

    # Optimized portfolio
    opt_weights = optimize(
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


if __name__ == "__main__":
    main()
