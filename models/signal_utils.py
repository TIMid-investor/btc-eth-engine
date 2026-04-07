"""
models/signal_utils.py — Shared signal processing utilities for crypto models.

Functions here are stateless transformations used by multiple models.
Import from here rather than duplicating in each model file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Compute trailing rolling Z-score: (x - rolling_mean) / rolling_std.

    Uses min_periods = window // 2 (minimum 14) to allow early signal formation
    without requiring a full window of history.

    Parameters
    ----------
    series : pd.Series with a DatetimeIndex
    window : lookback window in periods (days)

    Returns
    -------
    pd.Series of Z-scores; NaN where insufficient history or zero variance.
    """
    min_p = max(window // 2, 14)
    roll  = series.rolling(window=window, min_periods=min_p)
    mu    = roll.mean()
    sigma = roll.std()
    return (series - mu) / sigma.replace(0, np.nan)
