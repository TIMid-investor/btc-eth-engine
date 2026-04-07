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

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Single source of truth for demand weights lives in config.py.
# Editing weights: change config.DEMAND_WEIGHTS — do not add weights here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEMAND_WEIGHTS as DEFAULT_WEIGHTS
from models.signal_utils import rolling_zscore as _rolling_zscore

_NORM_WINDOW  = 90    # rolling Z-score window for each component
_SHORT_EMA    = 7     # demand_short EMA span
_TREND_EMA    = 30    # demand_trend EMA span
_CLIP_SIGMA   = 3.0   # clip normalized values to [-3, +3]


# ── Public API ─────────────────────────────────────────────────────────────────

def build_demand_index(
    trends_df:    pd.DataFrame | None = None,
    volume_df:    pd.DataFrame | None = None,
    etf_df:       pd.DataFrame | None = None,
    mvrv_df:      pd.DataFrame | None = None,
    outflows_df:  pd.DataFrame | None = None,
    fear_greed_df:pd.DataFrame | None = None,
    onchain_df:   pd.DataFrame | None = None,
    exchange_df:  pd.DataFrame | None = None,
    weights:      dict[str, float] | None = None,
    norm_window:  int = _NORM_WINDOW,
) -> pd.DataFrame:
    """
    Build the composite demand index from available data sources.

    Parameters
    ----------
    trends_df     : from ``fetch_trends_composite()`` — needs ``composite`` col
    volume_df     : OHLCV or CoinGecko df — needs ``volume`` (or ``total_volume``) col
    etf_df        : from ``fetch_etf_flows()`` — needs ``total_etf_volume`` col
    mvrv_df       : on-chain df with ``mvrv`` column (inverted: high MVRV = exit)
    outflows_df   : on-chain df with ``outflows`` column
    fear_greed_df : from ``fetch_fear_greed()`` — needs ``fear_greed`` col (0-100)
    onchain_df    : from ``build_onchain_frame()`` — used for active_addresses + mvrv
    exchange_df   : from ``fetch_exchange_volume()`` — needs ``total_exchange_volume``
    weights       : override default component weights (auto-renormalized)
    norm_window   : rolling Z-score window in days

    Returns
    -------
    DataFrame with columns:
      <component>_norm (each normalized input),
      demand_raw, demand_short, demand_trend, demand_rising
    """
    weights = dict(weights or DEFAULT_WEIGHTS)

    components: dict[str, pd.Series] = {}

    # ── A. Attention: Google Trends ────────────────────────────────────────────
    if trends_df is not None and "composite" in trends_df.columns:
        components["trends"] = trends_df["composite"].astype(float)
    else:
        weights["trends"] = 0.0

    # ── C. Action: Spot volume ─────────────────────────────────────────────────
    vol_series = None
    if volume_df is not None:
        if "total_volume" in volume_df.columns:
            vol_series = volume_df["total_volume"].astype(float)
        elif "volume" in volume_df.columns:
            vol_series = volume_df["volume"].astype(float)
            if "close" in volume_df.columns:
                vol_series = vol_series * volume_df["close"].astype(float)
    if vol_series is not None:
        components["volume"] = vol_series
    else:
        weights["volume"] = 0.0

    # ── B. Intent: MVRV (on-chain) — prefer onchain_df, fall back to mvrv_df ──
    mvrv_series = None
    if onchain_df is not None and "mvrv" in onchain_df.columns:
        mvrv_series = onchain_df["mvrv"].astype(float)
    elif mvrv_df is not None and "mvrv" in mvrv_df.columns:
        mvrv_series = mvrv_df["mvrv"].astype(float)
    if mvrv_series is not None:
        # Invert: high MVRV = overbought = low demand score
        components["mvrv"] = -mvrv_series
    else:
        weights["mvrv"] = 0.0

    # ── C. Action: ETF flows ───────────────────────────────────────────────────
    if etf_df is not None and "total_etf_volume" in etf_df.columns:
        components["etf"] = etf_df["total_etf_volume"].astype(float)
    else:
        weights["etf"] = 0.0

    # ── B. Intent: Exchange outflows ───────────────────────────────────────────
    if outflows_df is not None and "outflows" in outflows_df.columns:
        components["outflows"] = outflows_df["outflows"].astype(float)
    else:
        weights["outflows"] = 0.0

    # ── A. Attention: Fear & Greed Index ──────────────────────────────────────
    if fear_greed_df is not None and "fear_greed" in fear_greed_df.columns:
        # Use raw score (0-100); Z-score normalization handles scaling
        components["fear_greed"] = fear_greed_df["fear_greed"].astype(float)
    else:
        weights["fear_greed"] = 0.0

    # ── B. Intent: Active addresses (adoption proxy) ───────────────────────────
    addr_series = None
    if onchain_df is not None:
        for col in ("active_addresses", "btc_active_addresses", "eth_daily_tx_count"):
            if col in onchain_df.columns:
                addr_series = onchain_df[col].astype(float)
                break
    if addr_series is not None:
        components["active_addr"] = addr_series
    else:
        weights["active_addr"] = 0.0

    # ── C. Action: Multi-exchange aggregated volume ────────────────────────────
    if exchange_df is not None and "total_exchange_volume" in exchange_df.columns:
        components["exchange_vol"] = exchange_df["total_exchange_volume"].astype(float)
    else:
        weights["exchange_vol"] = 0.0

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

def _union_index(series_list: list[pd.Series]) -> pd.DatetimeIndex:
    """Return the union of all DatetimeIndexes, normalized to daily."""
    idx = series_list[0].index
    for s in series_list[1:]:
        idx = idx.union(s.index)
    return pd.DatetimeIndex(idx).normalize().sort_values()
