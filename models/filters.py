"""
models/filters.py — Optional confluence filters.

Each filter returns a boolean / directional Series that the signal
generator uses to gate or dampen entry signals.

  trend_filter      — discrete direction of intermediate trend (+1/0/-1)
  trend_multiplier  — continuous [0, 1] position-size multiplier based on
                      normalised EMA slope (sigmoid). Replaces the binary
                      on/off gate with a smooth gradient:
                        strong uptrend  → ~1.0 (full size)
                        flat trend      → ~0.82 (slight reduction)
                        strong downtrend → ~0.07 (almost zero)
  volume_filter   — weaken signals when volume is below its recent average
  macro_filter    — block signals during severe drawdowns (macro tail events)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Trend filter ───────────────────────────────────────────────────────────────

def trend_filter(prices: pd.Series, ema_days: int = 140) -> pd.Series:
    """
    Intermediate trend direction based on 20-week EMA slope.

    Returns:
       +1 when EMA slope is positive (uptrend)   — allow longs, block shorts
       -1 when EMA slope is negative (downtrend) — allow shorts, block longs
        0 on the first day (no prior bar)
    """
    ema = prices.ewm(span=ema_days, adjust=False).mean()
    slope = ema.diff()
    direction = np.sign(slope).fillna(0).astype(int)
    direction.name = "trend"
    return direction


def trend_multiplier(prices: pd.Series, ema_days: int = 140) -> pd.Series:
    """
    Continuous position-size multiplier based on EMA slope strength.

    Computes the fractional slope of the EMA, normalises it by its own
    rolling standard deviation (z-scores the slope), then maps through a
    sigmoid shifted so that a flat slope (slope_z = 0) gives ~0.82 rather
    than 0.5.  This means:

      strong uptrend   (slope_z ≈ +2) → multiplier ≈ 0.97  (nearly full size)
      flat trend       (slope_z ≈  0) → multiplier ≈ 0.82  (mild reduction)
      mild downtrend   (slope_z ≈ -1) → multiplier ≈ 0.67
      strong downtrend (slope_z ≈ -3) → multiplier ≈ 0.18  (heavily reduced)

    The shift of +1.5 in the exponent centres the "neutral" zone so that
    only meaningfully negative slopes shrink positions materially, while
    any positive slope leaves size largely intact.

    Returns Series in [0, 1] named "trend_mult".
    """
    ema = prices.ewm(span=ema_days, adjust=False).mean()
    frac_slope = ema.diff() / ema.shift(1)
    slope_std  = frac_slope.rolling(window=252, min_periods=30).std()
    slope_z    = frac_slope / slope_std.replace(0, np.nan)
    # Sigmoid shifted so slope_z = 0 → mult ≈ 0.82
    mult = 1.0 / (1.0 + np.exp(-(slope_z + 1.5)))
    return mult.clip(0, 1).fillna(0.5).rename("trend_mult")


# ── Volume filter ──────────────────────────────────────────────────────────────

def volume_filter(
    volume: pd.Series,
    window: int = 30,
    min_ratio: float = 0.80,
) -> pd.Series:
    """
    True when volume exceeds *min_ratio* × its rolling average.

    High deviation on low volume suggests a thin-market move rather than
    a genuine regime shift — lower conviction.

    Returns boolean Series named "volume_ok".
    """
    avg_vol = volume.rolling(window=window, min_periods=7).mean()
    volume_ok = (volume / avg_vol.replace(0, np.nan)) >= min_ratio
    volume_ok.name = "volume_ok"
    return volume_ok


# ── Macro / shock filter ───────────────────────────────────────────────────────

def macro_filter(
    prices: pd.Series,
    drawdown_threshold: float = -0.40,
    window: int = 90,
) -> pd.Series:
    """
    True when the asset is NOT in a severe macro drawdown.

    During a rapid -40%+ decline from a recent high the move is more
    likely driven by macro contagion (credit event, exchange collapse,
    etc.) than by mean-reversion around the long-run growth curve.
    Blocking trades in those periods reduces catching falling knives.

    Returns boolean Series named "macro_ok".
    """
    roll_high = prices.rolling(window=window, min_periods=1).max()
    drawdown  = prices / roll_high - 1.0
    macro_ok  = drawdown > drawdown_threshold
    macro_ok.name = "macro_ok"
    return macro_ok


# ── Composite filter ───────────────────────────────────────────────────────────

def build_filter_frame(
    prices: pd.Series,
    volume: pd.Series,
    cfg,
) -> pd.DataFrame:
    """
    Build a DataFrame with all filter columns using values from *cfg*.

    Parameters
    ----------
    prices  : close price Series
    volume  : volume Series
    cfg     : config module (or any object with the filter constants)

    Returns
    -------
    DataFrame with columns: trend, trend_mult, volume_ok, macro_ok
    """
    trend     = trend_filter(prices, ema_days=cfg.TREND_EMA_DAYS)
    volume_ok = volume_filter(volume, window=cfg.VOLUME_WINDOW,
                              min_ratio=cfg.VOLUME_MIN_RATIO)
    macro_ok  = macro_filter(prices, drawdown_threshold=cfg.MACRO_DD_THRESHOLD,
                             window=cfg.MACRO_DD_WINDOW)

    trend_mult = trend_multiplier(prices, ema_days=cfg.TREND_EMA_DAYS)

    return pd.DataFrame({
        "trend":      trend,
        "trend_mult": trend_mult,
        "volume_ok":  volume_ok,
        "macro_ok":   macro_ok,
    })
