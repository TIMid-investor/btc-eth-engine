"""
data/trends_fetcher.py — Google Trends data fetcher (attention proxy).

Pulls weekly interest-over-time from Google Trends via pytrends,
forward-fills to daily frequency, and caches to parquet.

Queries tracked:
  "bitcoin"      — broad awareness
  "buy bitcoin"  — purchase intent (leading retail signal)
  "crypto"       — category-level attention

Usage
-----
    from data.trends_fetcher import fetch_trends, fetch_trends_composite

    # Single query
    df = fetch_trends("bitcoin", start="2018-01-01")

    # Composite (all queries averaged, scaled 0-100)
    composite = fetch_trends_composite(start="2018-01-01")
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_CACHE

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_QUERIES = ["bitcoin", "buy bitcoin", "crypto"]

# Google Trends returns 0-100 scale natively; no further normalization needed.
# Weekly data is forward-filled to produce a daily series.

_CACHE_STALE_DAYS = 7   # re-fetch if last bar is older than this many days


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_trends(
    query: str,
    start: str = "2016-01-01",
    end: str | None = None,
    force_refresh: bool = False,
    geo: str = "",          # "" = worldwide; "US" = United States only
    retries: int = 3,
) -> pd.DataFrame:
    """
    Return daily Google Trends interest (0-100) for *query*.

    Columns: ``interest`` (float, 0-100 scale)
    Index  : DatetimeIndex (daily frequency, forward-filled from weekly source)

    Parameters
    ----------
    query         : search term, e.g. "bitcoin"
    start         : ISO date string for earliest bar to include
    end           : ISO date string (inclusive); defaults to today
    force_refresh : re-download even if cache is fresh
    geo           : geography code ("" = worldwide, "US" = United States)
    retries       : number of retry attempts on rate-limit / network errors
    """
    end = end or str(date.today())
    cache_path = _cache_path(query, geo)

    cached = _load_cache(cache_path)
    if cached is not None and not force_refresh:
        last_bar = cached.index.max()
        if (pd.Timestamp(end) - last_bar).days <= _CACHE_STALE_DAYS:
            return _filter_dates(cached, start, end)

    # Fetch from Google Trends
    df = _download_trends(query, start, end, geo, retries)
    if df.empty:
        if cached is not None:
            print(f"  [trends] download failed; using cached data for '{query}'",
                  flush=True)
            return _filter_dates(cached, start, end)
        raise RuntimeError(
            f"Could not fetch Google Trends data for '{query}'. "
            "Check that pytrends is installed: pip install pytrends>=4.9"
        )

    if cached is not None:
        # Merge: prefer fresh data over cached for overlapping dates
        combined = pd.concat([cached, df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        df = combined

    _save_cache(df, cache_path)
    return _filter_dates(df, start, end)


def fetch_trends_composite(
    queries: list[str] | None = None,
    start: str = "2016-01-01",
    end: str | None = None,
    geo: str = "",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a composite daily trends DataFrame combining multiple queries.

    Each query is fetched individually (pytrends scales each query 0-100
    independently, so direct averaging is appropriate for a composite index).

    Returned columns:
      - one column per query (e.g. ``bitcoin``, ``buy_bitcoin``, ``crypto``)
      - ``composite`` — equal-weight average of all available queries (0-100)

    Parameters
    ----------
    queries : list of search terms; defaults to DEFAULT_QUERIES
    start   : ISO date string
    end     : ISO date string; defaults to today
    geo     : geography code
    """
    queries = queries or DEFAULT_QUERIES
    end = end or str(date.today())

    frames: dict[str, pd.Series] = {}
    for q in queries:
        try:
            df = fetch_trends(q, start=start, end=end, geo=geo,
                              force_refresh=force_refresh)
            col_name = q.replace(" ", "_")
            frames[col_name] = df["interest"]
        except Exception as exc:
            print(f"  [trends] skipping '{q}': {exc}", flush=True)

    if not frames:
        raise RuntimeError("All Google Trends queries failed. Check pytrends install.")

    result = pd.DataFrame(frames)
    result["composite"] = result.mean(axis=1)
    result.index.name = "date"
    return result


# ── Internal helpers ───────────────────────────────────────────────────────────

def _download_trends(
    query: str,
    start: str,
    end: str,
    geo: str,
    retries: int,
) -> pd.DataFrame:
    """
    Download weekly trends from Google via pytrends, then forward-fill to daily.

    pytrends returns weekly data for ranges > 6 months.  We pull in
    6-month chunks to stay in the weekly-resolution regime, then stitch.
    """
    try:
        from pytrends.request import TrendReq  # type: ignore
    except ImportError:
        raise ImportError(
            "pytrends is not installed. Run: pip install pytrends>=4.9"
        )

    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)
    chunk_months = 6    # stay in weekly-resolution regime
    all_chunks: list[pd.DataFrame] = []

    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + pd.DateOffset(months=chunk_months), end_dt)
        timeframe = (
            f"{chunk_start.strftime('%Y-%m-%d')} "
            f"{chunk_end.strftime('%Y-%m-%d')}"
        )

        for attempt in range(retries):
            try:
                pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
                pytrends.build_payload(
                    kw_list=[query],
                    timeframe=timeframe,
                    geo=geo,
                )
                raw = pytrends.interest_over_time()
                if not raw.empty:
                    chunk_df = raw[[query]].rename(columns={query: "interest"})
                    chunk_df.index = pd.to_datetime(chunk_df.index).normalize()
                    chunk_df.index.name = "date"
                    all_chunks.append(chunk_df)
                break   # success — move to next chunk

            except Exception as exc:
                wait = 2 ** attempt * 3   # 3s, 6s, 12s
                print(
                    f"  [trends] '{query}' chunk {timeframe} attempt "
                    f"{attempt + 1}/{retries} failed: {exc}. "
                    f"Retrying in {wait}s…",
                    flush=True,
                )
                if attempt < retries - 1:
                    time.sleep(wait)

        # Polite pause between chunks to avoid 429 rate-limit
        time.sleep(1.5)
        chunk_start = chunk_end + timedelta(days=1)

    if not all_chunks:
        return pd.DataFrame()

    # Combine chunks, deduplicate, sort
    combined = pd.concat(all_chunks)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)

    # Forward-fill weekly values to daily frequency
    daily_idx = pd.date_range(
        start=combined.index.min(),
        end=combined.index.max(),
        freq="D",
    )
    daily = combined.reindex(daily_idx).ffill()
    daily.index.name = "date"
    return daily


def _cache_path(query: str, geo: str) -> Path:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    safe_query = query.replace(" ", "_").replace("/", "_")
    geo_suffix = f"_{geo}" if geo else ""
    return DATA_CACHE / f"trends_{safe_query}{geo_suffix}.parquet"


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
    try:
        df.to_parquet(path)
    except Exception as exc:
        print(f"  [trends] cache write failed: {exc}", flush=True)


def _filter_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask].copy()
