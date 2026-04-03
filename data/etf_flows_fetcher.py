"""
data/etf_flows_fetcher.py — Bitcoin ETF daily volume / flow proxy.

Pulls daily OHLCV for US-listed Bitcoin spot ETFs via yfinance (already a
project dependency — no new libraries needed).

ETFs tracked
------------
  IBIT  — iShares Bitcoin Trust (BlackRock, launched Jan 2024)
  FBTC  — Fidelity Wise Origin Bitcoin Fund (launched Jan 2024)
  ARKB  — ARK 21Shares Bitcoin ETF (launched Jan 2024)
  BITB  — Bitwise Bitcoin ETF (launched Jan 2024)
  HODL  — VanEck Bitcoin ETF (launched Jan 2024)
  GBTC  — Grayscale Bitcoin Trust (converted Jan 2024; had large outflows)

Flow proxy
----------
True AUM flows are disclosed by fund sponsors with a T+1 lag and require
scraping fund pages or paying for a data feed.  As a same-day proxy we use:

    dollar_volume  = close × volume           (total $ traded that day)
    flow_proxy     = dollar_volume × sign(close_chg)   (positive = net buy day)

This is an *approximation*.  It correlates well with actual flows on
high-conviction days but is noisier than real AUM data.

Usage
-----
    from data.etf_flows_fetcher import fetch_etf_flows

    df = fetch_etf_flows(start="2024-01-15")
    # Columns: ibit_vol, fbtc_vol, arkb_vol, bitb_vol, hodl_vol, gbtc_vol,
    #          total_etf_volume, total_etf_flow_proxy
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
    Return daily ETF volume and flow-proxy metrics.

    Columns
    -------
    <ticker>_vol        : daily dollar volume for each ETF (close × volume)
    <ticker>_flow       : signed flow proxy (positive = net buy day)
    total_etf_volume    : sum of all ETF dollar volumes
    total_etf_flow_proxy: sum of all signed flow proxies
    btc_etf_dominance   : IBIT + FBTC share of total ETF volume (institutional proxy)

    Parameters
    ----------
    start         : ISO date string (defaults to ETF launch date)
    end           : ISO date string; defaults to today
    force_refresh : re-download even if cache is fresh
    """
    end = end or str(date.today())
    cache_path = DATA_CACHE / _CACHE_FILE

    cached = _load_cache(cache_path)
    if cached is not None and not force_refresh:
        last_bar = cached.index.max()
        if (pd.Timestamp(end) - last_bar).days <= _CACHE_STALE_DAYS:
            return _filter_dates(cached, start, end)
        # Incremental refresh
        incremental_start = str((last_bar + timedelta(days=1)).date())
        fresh = _download_etfs(incremental_start, end)
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

def _download_etfs(start: str, end: str) -> pd.DataFrame:
    """Download OHLCV for all ETFs and compute flow proxy metrics."""
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
                # Single-ticker fallback (shouldn't happen with list input)
                close  = raw["Close"]
                volume = raw["Volume"]

            close  = close.squeeze()
            volume = volume.squeeze()

            # Dollar volume (flow magnitude)
            dollar_vol = (close * volume).rename(f"{col_name}_vol")

            # Signed flow proxy: positive on up-days, negative on down-days
            price_chg = close.pct_change().fillna(0)
            flow = (dollar_vol * np.sign(price_chg)).rename(f"{col_name}_flow")

            frames[f"{col_name}_vol"]  = dollar_vol
            frames[f"{col_name}_flow"] = flow

        except Exception as exc:
            print(f"  [etf_flows] skipping {ticker}: {exc}", flush=True)

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"

    # Aggregate columns
    vol_cols  = [c for c in df.columns if c.endswith("_vol")]
    flow_cols = [c for c in df.columns if c.endswith("_flow")]

    df["total_etf_volume"]     = df[vol_cols].sum(axis=1)
    df["total_etf_flow_proxy"] = df[flow_cols].sum(axis=1)

    # Institutional dominance proxy: BlackRock + Fidelity share of total volume
    inst_vol_cols = [c for c in ["ibit_vol", "fbtc_vol"] if c in df.columns]
    if inst_vol_cols and "total_etf_volume" in df.columns:
        total = df["total_etf_volume"].replace(0, np.nan)
        df["btc_etf_dominance"] = df[inst_vol_cols].sum(axis=1) / total
    else:
        df["btc_etf_dominance"] = np.nan

    return df.sort_index()


def _load_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "date"
        return df
    except Exception:
        return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except Exception as exc:
        print(f"  [etf_flows] cache write failed: {exc}", flush=True)


def _filter_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask].copy()
