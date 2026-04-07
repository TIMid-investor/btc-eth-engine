"""
data/cache_utils.py — Shared parquet cache helpers for all crypto data fetchers.

Every fetcher uses the same three-function pattern for reading, writing,
and date-filtering its parquet cache. Centralising here means any fix
(e.g. index normalisation, encoding issue) is applied once, everywhere.

Sidecar metadata (.meta.json)
------------------------------
``save_cache`` writes a tiny ``<name>.meta.json`` next to every parquet file:
    {"last_date": "2024-11-15", "rows": 2567}

``get_last_date`` reads only this sidecar — O(1), no parquet I/O — so
freshness checks in every fetcher avoid a full parquet scan just to get
the last date.

Usage
-----
    from data.cache_utils import load_cache as _load_cache
    from data.cache_utils import save_cache as _save_cache
    from data.cache_utils import filter_dates as _filter_dates
    from data.cache_utils import get_last_date as _get_last_date
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".meta.json")


def get_last_date(cache_path: Path) -> pd.Timestamp | None:
    """
    Return the last cached date in O(1) by reading the sidecar .meta.json.

    Returns ``None`` if the sidecar is missing (fall back to load_cache).
    """
    meta = _meta_path(cache_path)
    if not meta.exists():
        return None
    try:
        data = json.loads(meta.read_text())
        return pd.Timestamp(data["last_date"])
    except Exception:
        return None


def load_cache(path: Path) -> pd.DataFrame | None:
    """
    Read a parquet cache file.

    Returns a DataFrame with a normalised, tz-naive DatetimeIndex named
    ``"date"``, or ``None`` if the file is missing or unreadable.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "date"
        return df
    except Exception:
        return None


def save_cache(df: pd.DataFrame, path: Path) -> None:
    """
    Write *df* to *path* as parquet and update the companion .meta.json sidecar.

    Creating parent directories as needed. Silently prints a warning on
    write failure rather than raising, so a transient disk error does not
    abort the fetch pipeline.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        # Write sidecar so freshness checks are O(1)
        last_date = df.index.max()
        meta = {
            "last_date": str(last_date.date()) if hasattr(last_date, "date") else str(last_date)[:10],
            "rows": len(df),
        }
        _meta_path(path).write_text(json.dumps(meta))
    except Exception as exc:
        print(f"  [cache] write failed for {path.name}: {exc}", flush=True)


def filter_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Return a copy of *df* containing only rows where the index falls
    between *start* and *end* (both inclusive, ISO date strings).
    """
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask].copy()
