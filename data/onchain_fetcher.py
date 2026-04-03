"""
data/onchain_fetcher.py — On-chain metrics from free public APIs.

Three free data sources — no paid subscription required:

┌─────────────────┬──────────────────────────────────────────┬──────────┐
│ Source          │ Metrics                                  │ API key  │
├─────────────────┼──────────────────────────────────────────┼──────────┤
│ CoinMetrics     │ MVRV, realized cap, active addresses,    │ None     │
│ Community       │ tx volume, NVT                           │          │
├─────────────────┼──────────────────────────────────────────┼──────────┤
│ Blockchain.com  │ BTC active addresses, tx volume,         │ None     │
│                 │ hash rate (macro health proxy)           │          │
├─────────────────┼──────────────────────────────────────────┼──────────┤
│ Etherscan       │ ETH daily tx count, gas used             │ Free     │
│                 │ (register at etherscan.io)               │ ETHERSCAN│
│                 │                                          │ _API_KEY │
└─────────────────┴──────────────────────────────────────────┴──────────┘

MVRV Z-score (key signal)
-------------------------
  MVRV = Market Cap / Realized Cap
  MVRV Z-score = (MVRV - rolling_mean) / rolling_std
  Historically: >7 = cycle top, <0 = cycle bottom

Usage
-----
    from data.onchain_fetcher import fetch_coinmetrics, fetch_blockchain_info,
                                     fetch_etherscan, build_onchain_frame

    # All-in-one (preferred)
    df = build_onchain_frame("BTC", start="2016-01-01")
    # Columns: mvrv, mvrv_zscore, realized_cap, active_addresses,
    #          tx_volume_usd, nvt, exchange_inflow (BTC), exchange_outflow (BTC)
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_CACHE

# ── HTTP session ───────────────────────────────────────────────────────────────

_SESSION = requests.Session()
_SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "btc-eth-engine/1.0",
})

_TIMEOUT = 30


# ══════════════════════════════════════════════════════════════════════════════
# CoinMetrics Community API  (free, no key)
# ══════════════════════════════════════════════════════════════════════════════

_CM_BASE = "https://community-api.coinmetrics.io/v4"

# Map our asset symbol → CoinMetrics asset ID
_CM_ASSETS = {"BTC": "btc", "ETH": "eth"}

# CoinMetrics metric names we want
_CM_METRICS = {
    "CapMrktCurUSD":   "market_cap",       # Market cap (USD)
    "CapRealUSD":      "realized_cap",     # Realized cap (USD)
    "AdrActCnt":       "active_addresses", # Active addresses
    "TxTfrValAdjUSD":  "tx_volume_usd",    # Adjusted transfer volume (USD)
    "NVTAdj":          "nvt",              # Network Value to Transactions ratio
}


def fetch_coinmetrics(
    symbol: str,
    start: str = "2016-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch on-chain metrics from CoinMetrics Community API (free, no key).

    Returns daily DataFrame with columns:
        market_cap, realized_cap, active_addresses, tx_volume_usd, nvt,
        mvrv, mvrv_zscore
    """
    symbol = symbol.upper()
    if symbol not in _CM_ASSETS:
        raise ValueError(f"Unsupported symbol: {symbol}")

    end        = end or str(date.today())
    cache_path = DATA_CACHE / f"coinmetrics_{symbol}.parquet"

    cached = _load_cache(cache_path)
    if cached is not None and not force_refresh:
        last_bar = cached.index.max()
        if (pd.Timestamp(end) - last_bar).days <= 1:
            return _filter_dates(cached, start, end)
        incremental_start = str((last_bar + timedelta(days=1)).date())
        fresh = _download_coinmetrics(symbol, incremental_start, end)
        if fresh is not None and not fresh.empty:
            combined = pd.concat([cached, fresh])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            _save_cache(combined, cache_path)
            return _filter_dates(combined, start, end)
        return _filter_dates(cached, start, end)

    df = _download_coinmetrics(symbol, start, end)
    if df is None or df.empty:
        raise RuntimeError(
            f"CoinMetrics returned no data for {symbol}. "
            "Check connectivity at community-api.coinmetrics.io"
        )
    _save_cache(df, cache_path)
    return _filter_dates(df, start, end)


def _download_coinmetrics(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    asset    = _CM_ASSETS[symbol]
    metrics  = ",".join(_CM_METRICS.keys())

    params = {
        "assets":      asset,
        "metrics":     metrics,
        "start_time":  start,
        "end_time":    end,
        "frequency":   "1d",
        "page_size":   10_000,
    }

    data = _get_with_backoff(f"{_CM_BASE}/timeseries/asset-metrics", params)
    if data is None:
        return None

    rows = data.get("data", [])
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    df = df.set_index("date").drop(columns=["time", "asset"], errors="ignore")

    # Rename to friendly names
    rename = {k: v for k, v in _CM_METRICS.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Cast to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive MVRV and MVRV Z-score
    if "market_cap" in df.columns and "realized_cap" in df.columns:
        df["mvrv"] = df["market_cap"] / df["realized_cap"].replace(0, np.nan)
        # Rolling Z-score (365-day window)
        roll = df["mvrv"].rolling(365, min_periods=90)
        df["mvrv_zscore"] = (df["mvrv"] - roll.mean()) / roll.std()

    df.index.name = "date"
    return df.sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# Blockchain.com charts API  (BTC only, free, no key)
# ══════════════════════════════════════════════════════════════════════════════

_BC_BASE = "https://api.blockchain.info/charts"

_BC_CHARTS = {
    "n-unique-addresses":               "btc_active_addresses",
    "estimated-transaction-volume-usd": "btc_tx_volume_usd",
    "hash-rate":                        "btc_hash_rate",
}


def fetch_blockchain_info(
    start: str = "2016-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch BTC on-chain charts from Blockchain.com (free, no key required).

    Returns daily DataFrame with columns:
        btc_active_addresses, btc_tx_volume_usd, btc_hash_rate
    """
    end        = end or str(date.today())
    cache_path = DATA_CACHE / "blockchain_info.parquet"

    cached = _load_cache(cache_path)
    if cached is not None and not force_refresh:
        last_bar = cached.index.max()
        if (pd.Timestamp(end) - last_bar).days <= 1:
            return _filter_dates(cached, start, end)

    df = _download_blockchain_info(start, end)
    if df is None or df.empty:
        if cached is not None:
            print("  [blockchain.info] download failed; using cache", flush=True)
            return _filter_dates(cached, start, end)
        raise RuntimeError("Could not fetch Blockchain.com data.")

    if cached is not None:
        combined = pd.concat([cached, df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        df = combined

    _save_cache(df, cache_path)
    return _filter_dates(df, start, end)


def _download_blockchain_info(start: str, end: str) -> pd.DataFrame | None:
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts   = int(pd.Timestamp(end).timestamp())

    frames: dict[str, pd.Series] = {}
    for chart_name, col_name in _BC_CHARTS.items():
        params = {
            "timespan":  "all",
            "format":    "json",
            "cors":      "true",
            "start":     start_ts,
        }
        data = _get_with_backoff(f"{_BC_BASE}/{chart_name}", params)
        if data is None:
            continue
        values = data.get("values", [])
        if not values:
            continue
        dates = pd.to_datetime([v["x"] for v in values], unit="s").normalize()
        vals  = [float(v["y"]) for v in values]
        s = pd.Series(vals, index=dates, name=col_name)
        frames[col_name] = s
        time.sleep(0.8)   # polite pause between chart requests

    if not frames:
        return None

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df = df[df.index >= pd.Timestamp(start)]
    df = df[df.index <= pd.Timestamp(end)]
    return df.sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# Etherscan API  (ETH, free key — register at etherscan.io)
# ══════════════════════════════════════════════════════════════════════════════

_ES_BASE = "https://api.etherscan.io/api"


def fetch_etherscan(
    start: str = "2016-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame | None:
    """
    Fetch ETH on-chain metrics from Etherscan (free API key required).

    Set environment variable:  ETHERSCAN_API_KEY=<your_key>
    Register free at: https://etherscan.io/apis

    Returns daily DataFrame with columns:
        eth_daily_tx_count, eth_daily_gas_used

    Returns None (with a warning) if no API key is set.
    """
    api_key = os.environ.get("ETHERSCAN_API_KEY")
    if not api_key:
        print("  [etherscan] ETHERSCAN_API_KEY not set — skipping ETH on-chain metrics. "
              "Register free at etherscan.io/apis", flush=True)
        return None

    end        = end or str(date.today())
    cache_path = DATA_CACHE / "etherscan.parquet"

    cached = _load_cache(cache_path)
    if cached is not None and not force_refresh:
        last_bar = cached.index.max()
        if (pd.Timestamp(end) - last_bar).days <= 1:
            return _filter_dates(cached, start, end)

    df = _download_etherscan(api_key, start, end)
    if df is None or df.empty:
        if cached is not None:
            return _filter_dates(cached, start, end)
        return None

    if cached is not None:
        combined = pd.concat([cached, df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        df = combined

    _save_cache(df, cache_path)
    return _filter_dates(df, start, end)


def _download_etherscan(api_key: str, start: str, end: str) -> pd.DataFrame | None:
    start_unix = int(pd.Timestamp(start).timestamp())
    end_unix   = int(pd.Timestamp(end).timestamp())

    frames: dict[str, pd.Series] = {}

    # Daily transaction count
    params = {
        "module":    "stats",
        "action":    "dailytx",
        "startdate": start,
        "enddate":   end,
        "sort":      "asc",
        "apikey":    api_key,
    }
    data = _get_with_backoff(_ES_BASE, params)
    if data and data.get("status") == "1":
        rows = data.get("result", [])
        dates = pd.to_datetime([r["unixTimeStamp"] for r in rows], unit="s").normalize()
        vals  = [float(r["transactionCount"]) for r in rows]
        frames["eth_daily_tx_count"] = pd.Series(vals, index=dates)

    time.sleep(0.5)

    # Daily gas used
    params["action"] = "dailyavggasused"
    data = _get_with_backoff(_ES_BASE, params)
    if data and data.get("status") == "1":
        rows = data.get("result", [])
        dates = pd.to_datetime([r["unixTimeStamp"] for r in rows], unit="s").normalize()
        vals  = [float(r["gasUsed"]) for r in rows]
        frames["eth_daily_gas_used"] = pd.Series(vals, index=dates)

    if not frames:
        return None

    df = pd.DataFrame(frames)
    df.index.name = "date"
    return df.sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# All-in-one builder
# ══════════════════════════════════════════════════════════════════════════════

def build_onchain_frame(
    symbol: str,
    start: str = "2016-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Build a combined on-chain DataFrame from all available free sources.

    Tries each source independently; missing sources result in NaN columns.
    The demand index handles missing columns gracefully.

    Returns DataFrame with columns (subset depending on availability):
        mvrv, mvrv_zscore, realized_cap, active_addresses,
        tx_volume_usd, nvt,
        btc_active_addresses, btc_tx_volume_usd, btc_hash_rate,
        eth_daily_tx_count, eth_daily_gas_used
    """
    end = end or str(date.today())
    frames: list[pd.DataFrame] = []

    # CoinMetrics (both BTC and ETH)
    try:
        cm = fetch_coinmetrics(symbol, start=start, end=end,
                               force_refresh=force_refresh)
        frames.append(cm)
        print(f"  [onchain] CoinMetrics: {len(cm)} rows", flush=True)
    except Exception as exc:
        print(f"  [onchain] CoinMetrics failed: {exc}", flush=True)

    # Blockchain.com (BTC only)
    if symbol.upper() == "BTC":
        try:
            bc = fetch_blockchain_info(start=start, end=end,
                                       force_refresh=force_refresh)
            frames.append(bc)
            print(f"  [onchain] Blockchain.com: {len(bc)} rows", flush=True)
        except Exception as exc:
            print(f"  [onchain] Blockchain.com failed: {exc}", flush=True)

    # Etherscan (ETH only)
    if symbol.upper() == "ETH":
        try:
            es = fetch_etherscan(start=start, end=end,
                                 force_refresh=force_refresh)
            if es is not None:
                frames.append(es)
                print(f"  [onchain] Etherscan: {len(es)} rows", flush=True)
        except Exception as exc:
            print(f"  [onchain] Etherscan failed: {exc}", flush=True)

    if not frames:
        raise RuntimeError(f"All on-chain data sources failed for {symbol}.")

    # Combine on daily index (outer join — missing values stay NaN)
    result = frames[0]
    for f in frames[1:]:
        result = result.join(f, how="outer", rsuffix="_dup")
        # Drop any duplicated columns from rsuffix
        dup_cols = [c for c in result.columns if c.endswith("_dup")]
        result.drop(columns=dup_cols, inplace=True)

    result.sort_index(inplace=True)
    return result


# ── Shared HTTP helpers ────────────────────────────────────────────────────────

def _get_with_backoff(url: str, params: dict, retries: int = 4) -> dict | None:
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, params=params, timeout=_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  [onchain] rate-limited ({url}); waiting {wait}s…",
                      flush=True)
                time.sleep(wait)
                continue
            print(f"  [onchain] HTTP {resp.status_code} for {url}", flush=True)
            return None
        except requests.RequestException as exc:
            wait = 2 ** attempt * 3
            print(f"  [onchain] network error: {exc}; retry in {wait}s", flush=True)
            if attempt < retries - 1:
                time.sleep(wait)
    return None


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
        print(f"  [onchain] cache write failed: {exc}", flush=True)


def _filter_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask].copy()
