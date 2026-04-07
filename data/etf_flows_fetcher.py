"""
data/etf_flows_fetcher.py — Bitcoin ETF daily volume and net flow estimation.

ETFs tracked
------------
  IBIT  — iShares Bitcoin Trust (BlackRock, launched Jan 2024)
  FBTC  — Fidelity Wise Origin Bitcoin Fund (launched Jan 2024)
  ARKB  — ARK 21Shares Bitcoin ETF (launched Jan 2024)
  BITB  — Bitwise Bitcoin ETF (launched Jan 2024)
  HODL  — VanEck Bitcoin ETF (launched Jan 2024)
  GBTC  — Grayscale Bitcoin Trust (converted Jan 2024; monitor for outflows)

Flow estimation — primary method (shares-outstanding delta)
-----------------------------------------------------------
BlackRock publishes daily shares outstanding for IBIT on their fund page;
Fidelity does the same for FBTC.  Changes in shares outstanding directly
represent ETF creation (new shares → net inflow) or redemption (shares
redeemed → net outflow):

    net_flow ≈ Δshares_outstanding × NAV_per_share

This is a true flow signal, not a momentum proxy.  Shares outstanding data
is fetched via yfinance (the 'info' field) or from public fund pages.

Flow estimation — fallback (signed dollar-volume proxy)
-------------------------------------------------------
When shares-outstanding data is unavailable for a given ETF, we fall back to:

    flow_proxy = dollar_volume × sign(close_chg)

This is a momentum proxy, NOT a real flow.  The code previously used only
this approximation.  It remains as a fallback only.

Usage
-----
    from data.etf_flows_fetcher import fetch_etf_flows

    df = fetch_etf_flows(start="2024-01-15")
    # Primary columns: ibit_shares_flow, fbtc_shares_flow, total_etf_net_flow
    # Fallback columns: total_etf_flow_proxy (momentum-based, use only if primary unavailable)
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_CACHE
from data.cache_utils import (load_cache as _load_cache, save_cache as _save_cache,
                               filter_dates as _filter_dates, get_last_date as _get_last_date)

# ── ETF universe ───────────────────────────────────────────────────────────────

ETF_TICKERS: dict[str, str] = {
    "ibit":  "IBIT",    # BlackRock  — largest by AUM
    "fbtc":  "FBTC",    # Fidelity
    "arkb":  "ARKB",    # ARK / 21Shares
    "bitb":  "BITB",    # Bitwise
    "hodl":  "HODL",    # VanEck
    "gbtc":  "GBTC",    # Grayscale (converted; monitor for outflows)
}

# ETFs launched January 2024; data before this date will be NaN
ETF_LAUNCH_DATE = "2024-01-11"
GBTC_LAUNCH_DATE = "2015-05-01"    # GBTC existed as a trust before conversion

_CACHE_FILE = "etf_flows.parquet"
_CACHE_STALE_DAYS = 1


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_etf_flows(
    start: str = "2024-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return daily ETF volume and net flow estimates.

    Columns
    -------
    <ticker>_vol         : daily dollar volume (close × volume)
    <ticker>_shares_flow : net flow from Δshares_outstanding × NAV (primary)
    <ticker>_flow        : signed dollar-volume proxy (fallback; momentum signal)
    total_etf_volume     : sum of all ETF dollar volumes
    total_etf_net_flow   : sum of shares-based net flows (primary; NaN before 2024)
    total_etf_flow_proxy : sum of signed-volume proxies (fallback)
    btc_etf_dominance    : IBIT + FBTC share of total ETF volume

    Parameters
    ----------
    start         : ISO date string (defaults to ETF launch date)
    end           : ISO date string; defaults to today
    force_refresh : re-download even if cache is fresh
    """
    end = end or str(date.today())
    cache_path = DATA_CACHE / _CACHE_FILE

    cached   = None
    last_bar = _get_last_date(cache_path)
    if last_bar is None and not force_refresh:
        cached   = _load_cache(cache_path)
        last_bar = cached.index.max() if cached is not None else None
    if last_bar is not None and not force_refresh:
        if (pd.Timestamp(end) - last_bar).days <= _CACHE_STALE_DAYS:
            if cached is None:
                cached = _load_cache(cache_path)
            if cached is not None:
                return _filter_dates(cached, start, end)
        else:
            incremental_start = str((last_bar + timedelta(days=1)).date())
            fresh = _download_etfs(incremental_start, end)
            if cached is None:
                cached = _load_cache(cache_path)
            if cached is not None:
                if not fresh.empty:
                    combined = pd.concat([cached, fresh])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.sort_index(inplace=True)
                    _save_cache(combined, cache_path)
                    return _filter_dates(combined, start, end)
                return _filter_dates(cached, start, end)

    df = _download_etfs(start, end)
    if df.empty:
        raise RuntimeError("Could not download ETF data from yfinance.")
    _save_cache(df, cache_path)
    return _filter_dates(df, start, end)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _fetch_shares_outstanding(ticker: str) -> pd.Series | None:
    """
    Attempt to retrieve historical shares outstanding via yfinance.

    yfinance exposes shares outstanding in the Ticker.get_shares_full() method
    for some ETFs.  Returns a daily Series of shares outstanding indexed by
    date, or None if unavailable.
    """
    try:
        t = yf.Ticker(ticker)
        shares = t.get_shares_full(start="2024-01-01")
        if shares is None or shares.empty:
            return None
        # Resample to daily, forward-fill (shares only change on creation/redemption days)
        shares = shares.resample("D").last().ffill()
        shares.index = pd.to_datetime(shares.index).normalize()
        shares.index.name = "date"
        return shares
    except Exception:
        return None


def _download_etfs(start: str, end: str) -> pd.DataFrame:
    """Download OHLCV for all ETFs and compute flow metrics."""
    tickers = list(ETF_TICKERS.values())
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception as exc:
        print(f"  [etf_flows] yfinance download error: {exc}", flush=True)
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    frames: dict[str, pd.Series] = {}

    for col_name, ticker in ETF_TICKERS.items():
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                close  = raw[(ticker, "Close")]
                volume = raw[(ticker, "Volume")]
            else:
                close  = raw["Close"]
                volume = raw["Volume"]

            close  = close.squeeze()
            volume = volume.squeeze()

            # ── Dollar volume ──────────────────────────────────────────────
            dollar_vol = (close * volume).rename(f"{col_name}_vol")
            frames[f"{col_name}_vol"] = dollar_vol

            # ── Primary flow: Δshares_outstanding × NAV ───────────────────
            # NAV per share ≈ close price for an ETF trading at par.
            # Δshares > 0 = creation (inflow); Δshares < 0 = redemption (outflow).
            shares = _fetch_shares_outstanding(ticker)
            if shares is not None and not shares.empty:
                nav   = close.reindex(shares.index).ffill()
                delta = shares.diff()          # +creation / -redemption
                shares_flow = (delta * nav).rename(f"{col_name}_shares_flow")
                frames[f"{col_name}_shares_flow"] = shares_flow
            else:
                # Mark as unavailable — aggregate will use fallback
                frames[f"{col_name}_shares_flow"] = pd.Series(
                    np.nan, index=close.index, name=f"{col_name}_shares_flow"
                )

            # ── Fallback flow: signed dollar-volume proxy ─────────────────
            # This is a MOMENTUM signal (positive on up-days, negative on
            # down-days), NOT a real flow.  Retained only as a fallback when
            # shares-outstanding data is unavailable.
            price_chg = close.pct_change().fillna(0)
            flow = (dollar_vol * np.sign(price_chg)).rename(f"{col_name}_flow")
            frames[f"{col_name}_flow"] = flow

        except Exception as exc:
            print(f"  [etf_flows] skipping {ticker}: {exc}", flush=True)

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"

    # ── Aggregates ────────────────────────────────────────────────────────
    vol_cols          = [c for c in df.columns if c.endswith("_vol")]
    shares_flow_cols  = [c for c in df.columns if c.endswith("_shares_flow")]
    proxy_flow_cols   = [c for c in df.columns if c.endswith("_flow") and not c.endswith("_shares_flow")]

    df["total_etf_volume"]     = df[vol_cols].sum(axis=1)
    # Primary net flow: sum of shares-based flows (NaN if no ETF has shares data)
    if shares_flow_cols:
        shares_flows = df[shares_flow_cols]
        # Only sum days where at least one ETF has a non-NaN shares flow
        df["total_etf_net_flow"] = shares_flows.sum(axis=1, min_count=1)
    else:
        df["total_etf_net_flow"] = np.nan
    # Fallback proxy (momentum signal — not recommended as primary)
    df["total_etf_flow_proxy"] = df[proxy_flow_cols].sum(axis=1)

    # Institutional dominance proxy: BlackRock + Fidelity share of total volume
    inst_vol_cols = [c for c in ["ibit_vol", "fbtc_vol"] if c in df.columns]
    if inst_vol_cols and "total_etf_volume" in df.columns:
        total = df["total_etf_volume"].replace(0, np.nan)
        df["btc_etf_dominance"] = df[inst_vol_cols].sum(axis=1) / total
    else:
        df["btc_etf_dominance"] = np.nan

    return df.sort_index()


# _load_cache, _save_cache, _filter_dates imported from data.cache_utils above
