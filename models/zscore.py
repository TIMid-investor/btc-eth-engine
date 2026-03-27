"""
models/zscore.py — Rolling Z-score of log-deviations from the expected curve.

Z_t = log_deviation_t / rolling_std(log_deviation, window)

The rolling std uses only past observations (no look-ahead).
High Z → price above trend (overbought).
Low Z  → price below trend (oversold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(
    deviations: pd.Series,
    window: int = 365,
    min_periods: int = 180,
) -> pd.Series:
    """
    Compute rolling Z-score of the deviation series.

    Uses a *trailing* window so there is no look-ahead bias.
    Observations within `min_periods` of the start are set to NaN.

    Parameters
    ----------
    deviations  : log(price/expected) Series
    window      : rolling std window in days
    min_periods : minimum observations required before emitting a value

    Returns
    -------
    Series named "zscore"
    """
    rolling_std = deviations.rolling(window=window, min_periods=min_periods).std()
    zscore = deviations / rolling_std.replace(0, np.nan)
    zscore.name = "zscore"
    return zscore


def percentile_rank(deviations: pd.Series, window: int = 365) -> pd.Series:
    """
    Rolling percentile rank of the current deviation within the past *window* days.
    Returns values in [0, 1]; useful as an alternative signal to raw Z-score.
    """
    def rank_last(arr: np.ndarray) -> float:
        return float(np.sum(arr[:-1] < arr[-1]) / (len(arr) - 1)) if len(arr) > 1 else 0.5

    ranked = deviations.rolling(window=window, min_periods=30).apply(rank_last, raw=True)
    ranked.name = "pct_rank"
    return ranked
