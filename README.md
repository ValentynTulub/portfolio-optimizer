# Portfolio Optimizer

Pulls historical price data, computes risk metrics, and finds the weights
that maximize **Sortino**, **Sharpe**, or **Calmar** for a given basket of
tickers.

## Setup (Windows 11, Python 3.13)

Open PowerShell in the project folder and run:

```powershell
# Create a virtual environment
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1

# If activation is blocked by execution policy, run once as admin:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Install dependencies
pip install -e .
```

That installs `numpy`, `pandas`, `scipy`, and `yfinance`.

## Usage

Default run — your basket, last 10 years, optimized for Sortino:

```powershell
python optimize.py
```

With custom args:

```powershell
# 15 years of history, max 40% in any single asset
python optimize.py --years 15 --max-weight 0.40

# Different basket; note BRK.B is "BRK-B" in yfinance
python optimize.py 
VGT,VOO,QQQM,BRK-B,AMZN,GLD --years 10

# Optimize for Calmar instead (return / max drawdown)
python optimize.py --objective calmar

# Force every asset to be at least 5%
python optimize.py --min-weight 0.05 --max-weight 0.40
```

## What it shows

1. **Per-asset stats** — return, vol, Sharpe, Sortino, max drawdown for each ticker individually.
2. **Correlation matrix** — how assets move together. Look for low/negative correlations between holdings; that's where diversification value comes from.
3. **Equal-weight baseline** — the "no thinking" portfolio for comparison.
4. **Optimized portfolio** — weights that maximize your chosen objective, with full stats.

## Caveats worth re-reading

- The optimizer fits the *past*. If GLD had a great window, it'll suggest 30% gold. If VGT had a great window, it'll suggest 50% VGT. **Use `--max-weight` to cap concentration** (0.30–0.40 is sensible).
- **Try multiple time windows.** If the optimal weights are wildly different at 5 vs 10 vs 15 years, the signal is noisy. If they're stable, more trustworthy.
- **Sortino is one input, not the answer.** Consider qualitative factors: are you comfortable with this concentration? Do the weights make sense given your thesis?
- Risk-free rate hardcoded at 4% in `optimize.py`. Edit `RISK_FREE_RATE` if needed.

## Tickers gotcha

yfinance uses dashes, not dots, for share classes:
- BRK.B → `BRK-B`
- BF.B → `BF-B`
