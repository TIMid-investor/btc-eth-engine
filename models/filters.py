"""
models/filters.py — Optional confluence filters.

Each filter returns a boolean / directional Series that the signal
generator uses to gate or dampen entry signals.

  trend_filter    — only trade in the direction of the intermediate trend
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
    DataFrame with columns: trend, volume_ok, macro_ok
    """
    trend     = trend_filter(prices, ema_days=cfg.TREND_EMA_DAYS)
    volume_ok = volume_filter(volume, window=cfg.VOLUME_WINDOW,
                              min_ratio=cfg.VOLUME_MIN_RATIO)
    macro_ok  = macro_filter(prices, drawdown_threshold=cfg.MACRO_DD_THRESHOLD,
                             window=cfg.MACRO_DD_WINDOW)

    return pd.DataFrame({"trend": trend, "volume_ok": volume_ok, "macro_ok": macro_ok})
