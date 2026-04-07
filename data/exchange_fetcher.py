"""
data/exchange_fetcher.py — Multi-exchange volume via ccxt (free, no API key).

All endpoints used are *public* (market data only). No API key or account
required for OHLCV data on Binance, Coinbase, Kraken, or Bybit.

Metrics
-------
  Per-exchange daily dollar volume:
      binance_vol, coinbase_vol, kraken_vol, bybit_vol

  Coinbase Premium Index (institutional demand signal):
      coinbase_premium = BTC-USD(Coinbase) - BTC-USDT(Binance)
      Positive = US buyers paying up = institutional demand flowing in

  Aggregated:
      total_exchange_volume    (sum of all exchange $ volumes)
      coinbase_share           (Coinbase / total — US institutional proxy)

Usage
-----
    from data.exchange_fetcher import fetch_exchange_volume

    df = fetch_exchange_volume("BTC", start="2020-01-01")
    # Columns: binance_vol, coinbase_vol, kraken_vol, bybit_vol,
    #          total_exchange_volume, coinbase_premium, coinbase_share

Install
-------
    pip install ccxt>=4.3
"""

from __future__ import annotations

import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_CACHE
from data.cache_utils import (load_cache as _load_cache, save_cache as _save_cache,
                               filter_dates as _filter_dates, get_last_date as _get_last_date)

# ── Exchange configuration ─────────────────────────────────────────────────────

# (exchange_id, symbol, quote_currency_label)
# Binance uses USDT, Coinbase uses USD — price difference = Coinbase Premium
_EXCHANGE_CONFIGS = {
    "binance":  ("BTC/USDT", "ETH/USDT"),
    "coinbase": ("BTC/USD",  "ETH/USD"),
    "kraken":   ("BTC/USD",  "ETH/USD"),
    "bybit":    ("BTC/USDT", "ETH/USDT"),
}

_CACHE_STALE_DAYS = 1
_OHLCV_LIMIT      = 1000   # bars per ccxt fetch (max for most exchanges)
_MS_PER_DAY       = 86_400_000


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_exchange_volume(
    symbol: str = "BTC",
    start: str = "2018-01-01",
    end: str | None = None,
    force_refresh: bool = False,
    exchanges: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch multi-exchange daily dollar volume and Coinbase Premium Index.

    Parameters
    ----------
    symbol    : ``"BTC"`` or ``"ETH"``
    start     : ISO date string
    end       : ISO date string; defaults to today
    exchanges : subset of ["binance", "coinbase", "kraken", "bybit"]
                (default: all four)

    Returns
    -------
    DataFrame with columns (availability depends on ccxt install + network):
        <exchange>_vol, total_exchange_volume, coinbase_premium, coinbase_share
    """
    symbol    = symbol.upper()
    end       = end or str(date.today())
    exchanges = exchanges or list(_EXCHANGE_CONFIGS.keys())

    cache_path = DATA_CACHE / f"exchange_vol_{symbol}.parquet"

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
            inc_start = str((last_bar + pd.Timedelta(days=1)).date())
            fresh = _download_all_exchanges(symbol, inc_start, end, exchanges)
            if cached is None:
                cached = _load_cache(cache_path)
            if cached is not None:
                if fresh is not None and not fresh.empty:
                    combined = pd.concat([cached, fresh])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.sort_index(inplace=True)
                    _save_cache(combined, cache_path)
                    return _filter_dates(combined, start, end)
                return _filter_dates(cached, start, end)

    df = _download_all_exchanges(symbol, start, end, exchanges)
    if df is None or df.empty:
        raise RuntimeError(
            "Could not fetch exchange volume. "
            "Install ccxt: pip install ccxt>=4.3"
        )
    _save_cache(df, cache_path)
    return _filter_dates(df, start, end)


# ── Internal downloader ────────────────────────────────────────────────────────

def _download_all_exchanges(
    symbol: str,
    start: str,
    end: str,
    exchange_ids: list[str],
) -> pd.DataFrame | None:
    try:
        import ccxt  # type: ignore
    except ImportError:
        print(
            "  [exchange] ccxt not installed. Run: pip install ccxt>=4.3\n"
            "  [exchange] Skipping multi-exchange volume.",
            flush=True,
        )
        return None

    sym_idx   = 0 if symbol == "BTC" else 1
    start_ms  = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms    = int(pd.Timestamp(end).timestamp() * 1000)

    price_series: dict[str, pd.Series] = {}   # for Coinbase Premium
    vol_frames:   dict[str, pd.Series] = {}

    for ex_id in exchange_ids:
        if ex_id not in _EXCHANGE_CONFIGS:
            continue
        trading_pair = _EXCHANGE_CONFIGS[ex_id][sym_idx]

        try:
            exchange = getattr(ccxt, ex_id)({"enableRateLimit": True})
            ohlcv    = _fetch_ohlcv_chunked(exchange, trading_pair,
                                             start_ms, end_ms)
            if not ohlcv:
                continue

            df_ex = pd.DataFrame(
                ohlcv, columns=["ts", "open", "high", "low", "close", "volume"]
            )
            df_ex["date"]   = pd.to_datetime(df_ex["ts"], unit="ms").dt.normalize()
            df_ex           = df_ex.groupby("date").last().reset_index()
            df_ex           = df_ex.set_index("date")

            # Dollar volume = close × volume (in base currency units)
            dollar_vol = (df_ex["close"] * df_ex["volume"]).rename(f"{ex_id}_vol")
            vol_frames[f"{ex_id}_vol"] = dollar_vol

            # Keep close price for Coinbase Premium calculation
            if ex_id in ("binance", "coinbase"):
                price_series[ex_id] = df_ex["close"].rename(ex_id)

            print(f"  [exchange] {ex_id} {symbol}: {len(dollar_vol)} days", flush=True)
            time.sleep(0.5)   # polite between exchanges

        except Exception as exc:
            print(f"  [exchange] {ex_id} failed: {exc}", flush=True)

    fetched  = [ex for ex in exchange_ids if f"{ex}_vol" in vol_frames]
    failed   = [ex for ex in exchange_ids if f"{ex}_vol" not in vol_frames]
    if failed:
        print(f"  [exchange] {symbol} — fetched: {fetched or 'none'}  failed: {failed}", flush=True)

    if not vol_frames:
        return None

    result = pd.DataFrame(vol_frames)
    result.index.name = "date"

    # Total volume
    vol_cols = [c for c in result.columns if c.endswith("_vol")]
    result["total_exchange_volume"] = result[vol_cols].sum(axis=1)

    # Coinbase Premium Index (BTC only)
    if symbol == "BTC" and "binance" in price_series and "coinbase" in price_series:
        prices = pd.DataFrame(price_series)
        common = prices.index.intersection(result.index)
        premium = prices.loc[common, "coinbase"] - prices.loc[common, "binance"]
        result["coinbase_premium"] = premium
    else:
        result["coinbase_premium"] = np.nan

    # Coinbase share of total volume (US institutional proxy)
    if "coinbase_vol" in result.columns:
        total = result["total_exchange_volume"].replace(0, np.nan)
        result["coinbase_share"] = result["coinbase_vol"] / total
    else:
        result["coinbase_share"] = np.nan

    return result.sort_index()


def _fetch_ohlcv_chunked(
    exchange,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list:
    """Fetch all OHLCV bars between start_ms and end_ms in paginated chunks."""
    all_bars: list = []
    since = start_ms

    while since < end_ms:
        try:
            bars = exchange.fetch_ohlcv(
                symbol, timeframe="1d", since=since, limit=_OHLCV_LIMIT
            )
        except Exception as exc:
            print(f"  [exchange] fetch_ohlcv error ({symbol}): {exc}", flush=True)
            break

        if not bars:
            break

        all_bars.extend(bars)
        last_ts = bars[-1][0]

        if last_ts >= end_ms or len(bars) < _OHLCV_LIMIT:
            break

        since = last_ts + _MS_PER_DAY
        time.sleep(exchange.rateLimit / 1000.0)

    # Filter to requested range
    return [b for b in all_bars if start_ms <= b[0] <= end_ms]


# ── Cache helpers ──────────────────────────────────────────────────────────────
# _load_cache, _save_cache, _filter_dates imported from data.cache_utils above
