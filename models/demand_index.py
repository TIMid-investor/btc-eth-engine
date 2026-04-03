"""
models/demand_index.py — Composite demand index for BTC/ETH.

Combines multiple data sources into a single normalized score that
estimates *global marginal demand* for crypto.

Three demand buckets
--------------------
  A. Attention  — Google Trends (are people looking?)
  B. Intent     — on-chain accumulation / exchange outflows (are they buying?)
  C. Action     — ETF flows, spot volume (are they acting?)

Composite formula
-----------------
  demand_raw = w1*trends + w2*volume + w3*mvrv + w4*etf_flows + w5*outflows

Each component is first normalized to a rolling Z-score (90-day window)
so that all inputs are on a comparable scale regardless of unit.

Weights auto-renormalize when optional components are None.

Smoothing
---------
  demand_short = 7-day  EMA (noise reduction, tracks recent shifts)
  demand_trend = 30-day EMA (confirms sustained change)
  demand_rising = demand_short > demand_trend (Boolean)

Usage
-----
    from models.demand_index import build_demand_index

    demand_df = build_demand_index(
        trends_df  = composite_trends,   # from data.trends_fetcher
        volume_df  = ohlcv_df,           # close + volume columns
        etf_df     = etf_flows_df,       # from data.etf_flows_fetcher
        mvrv_df    = None,               # optional: from data.onchain_fetcher
        outflows_df= None,               # optional: from data.onchain_fetcher
    )
    # demand_df columns: demand_raw, demand_short, demand_trend, demand_rising,
    #                    <component>_norm (each normalized input for inspection)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Default weights (mirrored in config.py) ────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "trends":   0.25,   # Google Trends composite (attention)
    "volume":   0.25,   # Spot volume vs rolling avg (action)
    "mvrv":     0.20,   # MVRV Z-score (intent) — 0 if unavailable
    "etf":      0.20,   # ETF dollar volume (institutional action)
    "outflows": 0.10,   # Exchange outflows (accumulation intent) — 0 if unavailable
}

_NORM_WINDOW  = 90    # rolling Z-score window for each component
_SHORT_EMA    = 7     # demand_short EMA span
_TREND_EMA    = 30    # demand_trend EMA span
_CLIP_SIGMA   = 3.0   # clip normalized values to [-3, +3]


# ── Public API ─────────────────────────────────────────────────────────────────

def build_demand_index(
    trends_df:   pd.DataFrame | None = None,
    volume_df:   pd.DataFrame | None = None,
    etf_df:      pd.DataFrame | None = None,
    mvrv_df:     pd.DataFrame | None = None,
    outflows_df: pd.DataFrame | None = None,
    weights:     dict[str, float] | None = None,
    norm_window: int = _NORM_WINDOW,
) -> pd.DataFrame:
    """
    Build the composite demand index from available data sources.

    Parameters
    ----------
    trends_df   : DataFrame from ``fetch_trends_composite()``.
                  Must have a ``composite`` column (0-100 scale, daily).
    volume_df   : OHLCV DataFrame from ``fetch_ohlcv()``.
                  Must have ``close`` and ``volume`` columns.
    etf_df      : DataFrame from ``fetch_etf_flows()``.
                  Must have ``total_etf_volume`` column.
    mvrv_df     : Optional. DataFrame with ``mvrv`` column (on-chain).
    outflows_df : Optional. DataFrame with ``outflows`` column (on-chain).
    weights     : Override default component weights.  Will be renormalized
                  to sum to 1.0 automatically.
    norm_window : Rolling window (days) for Z-score normalization.

    Returns
    -------
    DataFrame with columns:
      trends_norm, volume_norm, mvrv_norm, etf_norm, outflows_norm
      demand_raw, demand_short, demand_trend, demand_rising
    """
    weights = dict(weights or DEFAULT_WEIGHTS)

    components: dict[str, pd.Series] = {}

    # ── A. Attention: Google Trends ────────────────────────────────────────────
    if trends_df is not None and "composite" in trends_df.columns:
        components["trends"] = trends_df["composite"].astype(float)
    else:
        weights["trends"] = 0.0

    # ── C. Action: Spot volume (volume ratio vs 90-day rolling avg) ────────────
    if volume_df is not None and "volume" in volume_df.columns:
        vol = volume_df["volume"].astype(float)
        # Dollar volume preferred; fall back to raw volume
        if "close" in volume_df.columns:
            vol = vol * volume_df["close"].astype(float)
        components["volume"] = vol
    else:
        weights["volume"] = 0.0

    # ── B. Intent: MVRV Z-score (on-chain, optional) ──────────────────────────
    if mvrv_df is not None and "mvrv" in mvrv_df.columns:
        # MVRV is already a ratio-like metric; we'll normalize it like others.
        # Invert sign: high MVRV = overbought = LOW demand score
        components["mvrv"] = -mvrv_df["mvrv"].astype(float)
    else:
        weights["mvrv"] = 0.0

    # ── C. Action: ETF flows (institutional, optional) ────────────────────────
    if etf_df is not None and "total_etf_volume" in etf_df.columns:
        components["etf"] = etf_df["total_etf_volume"].astype(float)
    else:
        weights["etf"] = 0.0

    # ── B. Intent: Exchange outflows (accumulation proxy, optional) ───────────
    if outflows_df is not None and "outflows" in outflows_df.columns:
        components["outflows"] = outflows_df["outflows"].astype(float)
    else:
        weights["outflows"] = 0.0

    if not components:
        raise ValueError(
            "At least one data source must be provided to build the demand index."
        )

    # ── Renormalize weights to sum to 1.0 ──────────────────────────────────────
    active_keys  = [k for k in components]
    active_total = sum(weights.get(k, 0.0) for k in active_keys)
    if active_total <= 0:
        raise ValueError("All component weights are zero.")
    norm_weights = {k: weights.get(k, 0.0) / active_total for k in active_keys}

    # ── Normalize each component to rolling Z-score ────────────────────────────
    # Align all series to a common daily DatetimeIndex (union of all indices)
    all_dates = _union_index(list(components.values()))

    result = pd.DataFrame(index=all_dates)
    result.index.name = "date"

    demand_raw = pd.Series(0.0, index=all_dates)

    for key, raw_series in components.items():
        aligned = raw_series.reindex(all_dates)
        normed  = _rolling_zscore(aligned, window=norm_window)
        normed  = normed.clip(-_CLIP_SIGMA, _CLIP_SIGMA)
        result[f"{key}_norm"] = normed
        demand_raw = demand_raw + norm_weights[key] * normed.fillna(0.0)

    result["demand_raw"]   = demand_raw

    # ── Smoothing ──────────────────────────────────────────────────────────────
    result["demand_short"] = demand_raw.ewm(span=_SHORT_EMA, adjust=False).mean()
    result["demand_trend"] = demand_raw.ewm(span=_TREND_EMA, adjust=False).mean()
    result["demand_rising"] = (result["demand_short"] > result["demand_trend"]).astype(int)

    return result


# ── Convenience: quick summary for the signal dashboard ───────────────────────

def demand_summary(demand_df: pd.DataFrame, as_of: str | None = None) -> dict:
    """
    Return a dict with current demand index values for display.

    Keys: demand_raw, demand_short, demand_trend, demand_rising,
          component norms available (e.g. trends_norm, volume_norm)
    """
    as_of = as_of or str(demand_df.index.max().date())
    row = demand_df.loc[:as_of].iloc[-1]
    return row.to_dict()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Compute trailing rolling Z-score: (x - rolling_mean) / rolling_std.
    Uses min_periods = window // 2 to allow early signal formation.
    """
    min_p = max(window // 2, 14)
    roll  = series.rolling(window=window, min_periods=min_p)
    mu    = roll.mean()
    sigma = roll.std()
    z = (series - mu) / sigma.replace(0, np.nan)
    return z


def _union_index(series_list: list[pd.Series]) -> pd.DatetimeIndex:
    """Return the union of all DatetimeIndexes, normalized to daily."""
    idx = series_list[0].index
    for s in series_list[1:]:
        idx = idx.union(s.index)
    return pd.DatetimeIndex(idx).normalize().sort_values()
