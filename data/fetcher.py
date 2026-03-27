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

    cached = _load_cache(cache_path)
    if cached is not None and not force_refresh:
        last_bar = cached.index.max()
        # Refresh if last bar is more than 1 calendar day old
        if (pd.Timestamp(end) - last_bar).days <= 1:
            return _filter_dates(cached, start, end)
        # Incremental update: fetch only the missing tail
        incremental_start = str((last_bar + timedelta(days=1)).date())
        fresh = _download(symbol, incremental_start, end)
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


def _load_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index).normalize()
        return df
    except Exception:
        return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path)
    except Exception as exc:
        print(f"  [fetcher] cache write failed: {exc}", flush=True)


def _filter_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask].copy()
