"""
data/coingecko_fetcher.py — CoinGecko market data fetcher.

Pulls daily price, volume, and market cap from the CoinGecko public API
(free tier, no API key required for basic endpoints).

Metrics captured
----------------
  close        : daily close price (USD)
  total_volume : total global spot volume (USD)
  market_cap   : market capitalisation (USD)
  volume_ratio : total_volume / market_cap  (liquidity signal; spikes at extremes)

BTC dominance
-------------
  Pulled from /global endpoint: btc_dominance (%), eth_dominance (%)
  Rising BTC dominance = risk-off / flight to quality (bearish for alts)
  Falling BTC dominance = risk-on / alt season

Rate limits
-----------
  Free tier: ~10-30 req/min.  We add time.sleep(1.5) between calls and
  use exponential backoff on 429 errors.

Usage
-----
    from data.coingecko_fetcher import fetch_coingecko, fetch_dominance

    df  = fetch_coingecko("bitcoin", start="2018-01-01")
    dom = fetch_dominance(start="2018-01-01")
"""

from __future__ import annotations

import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_CACHE
from data.cache_utils import (load_cache as _load_cache, save_cache as _save_cache,
                               filter_dates as _filter_dates, get_last_date as _get_last_date)

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_URL  = "https://api.coingecko.com/api/v3"
COIN_IDS  = {"BTC": "bitcoin", "ETH": "ethereum"}

# Days per chunk for historical pulls (CoinGecko returns daily OHLCV for
# ranges > 90 days automatically; we chunk to stay within free-tier page size)
_CHUNK_DAYS      = 365
_CACHE_STALE_DAYS = 1
_REQUEST_TIMEOUT  = 20   # seconds

_SESSION = requests.Session()
_SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "btc-eth-engine/1.0",
})


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_coingecko(
    symbol: str,
    start: str = "2016-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return daily CoinGecko market data for *symbol* (``"BTC"`` or ``"ETH"``).

    Columns
    -------
    close, total_volume, market_cap, volume_ratio

    Parameters
    ----------
    symbol        : ``"BTC"`` or ``"ETH"``
    start         : ISO date string
    end           : ISO date string; defaults to today
    force_refresh : re-download even if cache is fresh
    """
    symbol = symbol.upper()
    if symbol not in COIN_IDS:
        raise ValueError(f"Unsupported symbol '{symbol}'. Use 'BTC' or 'ETH'.")

    coin_id = COIN_IDS[symbol]
    end     = end or str(date.today())
    cache_path = DATA_CACHE / f"coingecko_{symbol}.parquet"

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
            fresh = _download_market_chart(coin_id, incremental_start, end)
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

    df = _download_market_chart(coin_id, start, end)
    if df.empty:
        raise RuntimeError(
            f"CoinGecko returned no data for {symbol}. "
            "Check connectivity or try again later."
        )
    _save_cache(df, cache_path)
    return _filter_dates(df, start, end)


def fetch_dominance(
    start: str = "2016-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return daily BTC and ETH dominance percentages from CoinGecko /global.

    Columns: btc_dominance, eth_dominance

    Note: CoinGecko /global only returns the *current* value, not history.
    This function caches today's reading and accumulates it over time.
    For historical dominance, the BTC market cap / total crypto market cap
    ratio is a better approach (available in the coin-level market chart).
    """
    end        = end or str(date.today())
    cache_path = DATA_CACHE / "coingecko_dominance.parquet"

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

    if cached is None:
        cached = _load_cache(cache_path)

    # Fetch current dominance and append to cache
    today_row = _fetch_current_dominance()
    if today_row is not None:
        if cached is not None:
            combined = pd.concat([cached, today_row])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
        else:
            combined = today_row
        _save_cache(combined, cache_path)
        return _filter_dates(combined, start, end)

    if cached is not None:
        return _filter_dates(cached, start, end)

    raise RuntimeError("Could not fetch CoinGecko dominance data.")


# ── Internal: market chart ─────────────────────────────────────────────────────

def _download_market_chart(coin_id: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from /coins/{id}/market_chart/range in chunks."""
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)
    all_chunks: list[pd.DataFrame] = []

    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS), end_dt)

        from_ts = int(chunk_start.timestamp())
        to_ts   = int(chunk_end.timestamp())

        url = f"{BASE_URL}/coins/{coin_id}/market_chart/range"
        params = {"vs_currency": "usd", "from": from_ts, "to": to_ts}

        data = _get_with_backoff(url, params)
        if data is None:
            chunk_start = chunk_end + timedelta(days=1)
            continue

        chunk_df = _parse_market_chart(data)
        if not chunk_df.empty:
            all_chunks.append(chunk_df)

        # Polite pause between chunks
        time.sleep(1.5)
        chunk_start = chunk_end + timedelta(days=1)

    if not all_chunks:
        return pd.DataFrame()

    result = pd.concat(all_chunks)
    result = result[~result.index.duplicated(keep="last")]
    result.sort_index(inplace=True)
    return result


def _parse_market_chart(data: dict) -> pd.DataFrame:
    """Parse CoinGecko market_chart/range response into a daily DataFrame."""
    prices      = data.get("prices", [])
    volumes     = data.get("total_volumes", [])
    market_caps = data.get("market_caps", [])

    if not prices:
        print("  [coingecko] _parse_market_chart: response contained no price data", flush=True)
        return pd.DataFrame()

    def _to_series(raw: list, name: str) -> pd.Series:
        ts = pd.to_datetime([r[0] for r in raw], unit="ms", utc=True)
        vals = [r[1] for r in raw]
        return pd.Series(vals, index=ts, name=name)

    price_s  = _to_series(prices,      "close")
    vol_s    = _to_series(volumes,     "total_volume")
    mcap_s   = _to_series(market_caps, "market_cap")

    df = pd.DataFrame({"close": price_s, "total_volume": vol_s, "market_cap": mcap_s})
    df.index = df.index.normalize().tz_localize(None)
    df.index.name = "date"

    # Resample to daily (CoinGecko sometimes returns multiple ticks per day)
    df = df.resample("D").last().dropna(how="all")

    # Volume-to-market-cap ratio (spikes at tops/bottoms)
    df["volume_ratio"] = df["total_volume"] / df["market_cap"].replace(0, float("nan"))

    return df


# ── Internal: dominance ────────────────────────────────────────────────────────

def _fetch_current_dominance() -> pd.DataFrame | None:
    """Fetch current BTC/ETH dominance from /global."""
    data = _get_with_backoff(f"{BASE_URL}/global", {})
    if data is None:
        return None
    try:
        d = data["data"]["market_cap_percentage"]
        today = pd.Timestamp(date.today())
        return pd.DataFrame(
            {"btc_dominance": d.get("btc", float("nan")),
             "eth_dominance": d.get("eth", float("nan"))},
            index=[today],
        )
    except Exception:
        return None


# ── Internal: HTTP helpers ─────────────────────────────────────────────────────

def _get_with_backoff(url: str, params: dict, retries: int = 4) -> dict | None:
    """GET with exponential backoff on 429 / network errors."""
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 2 ** attempt * 5   # 5s, 10s, 20s, 40s
                print(f"  [coingecko] rate-limited; waiting {wait}s…", flush=True)
                time.sleep(wait)
                continue
            print(f"  [coingecko] HTTP {resp.status_code} for {url}", flush=True)
            return None
        except requests.RequestException as exc:
            wait = 2 ** attempt * 3
            print(f"  [coingecko] network error ({exc}); retrying in {wait}s…",
                  flush=True)
            if attempt < retries - 1:
                time.sleep(wait)
    return None


# _load_cache, _save_cache, _filter_dates imported from data.cache_utils above
