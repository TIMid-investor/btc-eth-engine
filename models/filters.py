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
    dd_soft: float = -0.20,
    dd_hard: float = -0.50,
    window: int = 90,
) -> pd.Series:
    """
    Continuous position-size multiplier based on asset drawdown from recent high.

    Returns a Series in [0.0, 1.0] named "macro_mult":
      - 1.0  when drawdown is above dd_soft  (no suppression)
      - 0.0  when drawdown is at or below dd_hard (fully suppressed)
      - linearly interpolated in between

    This replaces the previous binary gate at -40% which would have nearly
    blocked the March 2020 BTC entry (drawdown ~-41% on the signal day).
    A continuous scale-down is more robust to threshold sensitivity.

    Parameters
    ----------
    prices   : asset close price Series
    dd_soft  : drawdown level at which scaling starts (e.g. -0.20 = -20%)
    dd_hard  : drawdown level at which position reaches zero (e.g. -0.50 = -50%)
    window   : rolling window for computing the high
    """
    roll_high = prices.rolling(window=window, min_periods=1).max()
    drawdown  = prices / roll_high - 1.0          # 0.0 at high, negative below
    # Linear interpolation: 1.0 at dd_soft, 0.0 at dd_hard
    span      = dd_hard - dd_soft                  # negative span
    mult      = ((drawdown - dd_soft) / span).clip(0.0, 1.0)
    # Invert: high drawdown → low mult
    macro_mult = 1.0 - mult
    macro_mult.name = "macro_mult"
    return macro_mult


# ── Demand filters ────────────────────────────────────────────────────────────

def demand_entry_filter(demand_rising: pd.Series) -> pd.Series:
    """
    True when short-term demand momentum is above its long-term trend.

    Only allow new entries on days when demand_short > demand_trend
    (i.e., demand is accelerating, not decelerating).

    Parameters
    ----------
    demand_rising : Boolean / int Series from build_demand_index()
                    (1 = rising, 0 = falling)

    Returns boolean Series named "demand_entry_ok".
    """
    result = demand_rising.astype(bool)
    result.name = "demand_entry_ok"
    return result


def demand_exit_filter(
    demand_short: pd.Series,
    demand_trend: pd.Series,
    peak_lookback: int = 10,
) -> pd.Series:
    """
    True when demand has peaked and is rolling over — signals early exit.

    Detection: demand_short was above demand_trend for >= `peak_lookback`
    consecutive days, and has now crossed below it (momentum exhaustion).

    Parameters
    ----------
    demand_short  : short-term smoothed demand (7-day EMA)
    demand_trend  : long-term smoothed demand (30-day EMA)
    peak_lookback : minimum days demand must have been rising before
                    a rollover counts as a peak signal

    Returns boolean Series named "demand_rolling_over" (True = exit signal).
    """
    rising = (demand_short > demand_trend).astype(int)

    # Count consecutive rising days
    consecutive = rising.groupby((rising != rising.shift()).cumsum()).cumcount() + 1
    consecutive[rising == 0] = 0

    # A peak rollover: was rising for >= peak_lookback days, now just crossed below
    just_crossed_below = (rising == 0) & (rising.shift(1) == 1)
    had_enough_history  = consecutive.shift(1).fillna(0) >= peak_lookback

    rolling_over = just_crossed_below & had_enough_history
    rolling_over.name = "demand_rolling_over"
    return rolling_over


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
    # Support both old binary threshold config and new soft/hard config.
    dd_soft = getattr(cfg, "MACRO_DD_SOFT", getattr(cfg, "MACRO_DD_THRESHOLD", -0.20))
    dd_hard = getattr(cfg, "MACRO_DD_HARD", -0.50)
    macro_mult = macro_filter(prices, dd_soft=dd_soft, dd_hard=dd_hard,
                              window=cfg.MACRO_DD_WINDOW)

    trend_mult = trend_multiplier(prices, ema_days=cfg.TREND_EMA_DAYS)

    return pd.DataFrame({
        "trend":      trend,
        "trend_mult": trend_mult,
        "volume_ok":  volume_ok,
        "macro_ok":   macro_mult >= 0.01,  # boolean alias kept for compatibility
        "macro_mult": macro_mult,
    })
