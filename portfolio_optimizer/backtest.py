"""Walk-forward (out-of-sample) testing utilities.

The optimizer fits historical data — by construction it overweights whatever
won in the chosen window. Splitting the data into a fit window and an
unseen-by-the-optimizer tail and re-evaluating the fitted weights on that tail
shows how much of the in-sample edge survives in genuinely new data.
"""

from __future__ import annotations

import pandas as pd


def split_returns(
    returns: pd.DataFrame, n_oos_years: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split daily returns into a (train, test) pair.

    The test window contains the last `n_oos_years` of data. The train window
    is everything before. Caller is responsible for checking each side has
    enough rows for meaningful statistics.
    """
    if n_oos_years <= 0:
        raise ValueError("n_oos_years must be positive")
    if returns.empty:
        raise ValueError("returns is empty")

    cutoff = returns.index.max() - pd.DateOffset(years=n_oos_years)
    train = returns.loc[returns.index <= cutoff]
    test = returns.loc[returns.index > cutoff]
    return train, test
