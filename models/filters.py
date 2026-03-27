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
