"""
data/fetcher.py — Download and cache OHLCV data from yfinance.

Parquet files are written to DATA_CACHE / <symbol>.parquet.
On subsequent calls the cached file is refreshed only if it is stale
(last bar is more than 1 day old), so repeated runs are fast.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_CACHE
from data.cache_utils import (load_cache as _load_cache, save_cache as _save_cache,
                               filter_dates as _filter_dates, get_last_date as _get_last_date)


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    start: str = "2014-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return daily OHLCV for *symbol* as a DataFrame indexed by date.

    Columns: open, high, low, close, volume
    Data is cached to DATA_CACHE/<symbol>.parquet.

    Parameters
    ----------
    symbol        : yfinance ticker, e.g. "BTC-USD"
    start         : ISO date string for the earliest bar to include
    end           : ISO date string (inclusive); defaults to today
    force_refresh : re-download even if cache is fresh
    """
    end = end or str(date.today())
    cache_path = _cache_path(symbol)

    # Sidecar-first freshness check — avoids parquet read for date lookup
    cached   = None
    last_bar = _get_last_date(cache_path)
    if last_bar is None and not force_refresh:
        cached   = _load_cache(cache_path)
        last_bar = cached.index.max() if cached is not None else None
    if last_bar is not None and not force_refresh:
        if (pd.Timestamp(end) - last_bar).days <= 1:
            if cached is None:
                cached = _load_cache(cache_path)
            if cached is not None:
                return _filter_dates(cached, start, end)
        else:
            # Incremental update: fetch only the missing tail
            incremental_start = str((last_bar + timedelta(days=1)).date())
            fresh = _download(symbol, incremental_start, end)
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

    # Full download
    df = _download(symbol, start, end)
    if df.empty:
        raise RuntimeError(f"No data returned from yfinance for {symbol}")
    _save_cache(df, cache_path)
    return _filter_dates(df, start, end)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _cache_path(symbol: str) -> Path:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("-", "_").replace("/", "_")
    return DATA_CACHE / f"{safe}.parquet"


def _download(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download from yfinance; return normalised DataFrame or empty."""
    try:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        print(f"  [fetcher] yfinance error for {symbol}: {exc}", flush=True)
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    df = df[df["close"] > 0].dropna(subset=["close"])
    return df


# _load_cache, _save_cache, _filter_dates imported from data.cache_utils above
