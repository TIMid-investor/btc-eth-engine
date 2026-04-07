#!/usr/bin/env python3
"""
scripts/fetch_data.py — Daily crypto data fetch orchestrator.

Called by pipeline/run_all.sh as part of the unified daily pipeline.
Refreshes all crypto data sources using the same incremental caching
pattern as each underlying fetcher (staleness-aware, idempotent).

Exit codes:
  0 — core OHLCV fetch succeeded (other sources may have warned)
  1 — core OHLCV fetch failed (pipeline should treat crypto as unavailable)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add projects/crypto/ to path so data.* imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _run(label: str, fn, *args, **kwargs) -> bool:
    """Call fn(*args, **kwargs); return True on success, False on failure."""
    _log(f"  {label} ...")
    try:
        fn(*args, **kwargs)
        _log(f"  {label} OK")
        return True
    except Exception as exc:
        _log(f"  {label} WARNING: {exc}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    _log("=========================================")
    _log("Crypto data fetch starting")
    _log("=========================================")

    # ── 1. Core OHLCV — BTC + ETH (critical) ──────────────────────────────────
    from data.fetcher import fetch_ohlcv

    btc_ok = _run("BTC-USD OHLCV", fetch_ohlcv, "BTC-USD")
    eth_ok = _run("ETH-USD OHLCV", fetch_ohlcv, "ETH-USD")

    if not (btc_ok and eth_ok):
        _log("FAILED: core OHLCV fetch failed — aborting")
        return 1

    # ── 2. CoinGecko price/volume/dominance (non-fatal) ───────────────────────
    try:
        from data.coingecko_fetcher import fetch_coingecko, fetch_dominance
        _run("CoinGecko BTC", fetch_coingecko, "BTC")
        _run("CoinGecko ETH", fetch_coingecko, "ETH")
        _run("BTC dominance", fetch_dominance)
    except ImportError as exc:
        _log(f"  CoinGecko skipped (import error): {exc}")

    # ── 3. Exchange volumes — Binance / Coinbase / Kraken (non-fatal) ─────────
    try:
        from data.exchange_fetcher import fetch_exchange_volume
        _run("Exchange volume BTC/USDT", fetch_exchange_volume, "BTC/USDT")
        _run("Exchange volume ETH/USDT", fetch_exchange_volume, "ETH/USDT")
    except ImportError as exc:
        _log(f"  Exchange volume skipped (import error): {exc}")

    # ── 4. On-chain metrics (non-fatal) ───────────────────────────────────────
    try:
        from data.onchain_fetcher import fetch_coinmetrics, fetch_blockchain_info
        _run("On-chain BTC (CoinMetrics)", fetch_coinmetrics, "BTC")
        _run("On-chain BTC (Blockchain.com)", fetch_blockchain_info)
    except ImportError as exc:
        _log(f"  On-chain skipped (import error): {exc}")

    # ── 5. BTC ETF net flows (non-fatal) ──────────────────────────────────────
    try:
        from data.etf_flows_fetcher import fetch_etf_flows
        _run("BTC ETF flows", fetch_etf_flows)
    except ImportError as exc:
        _log(f"  ETF flows skipped (import error): {exc}")

    # ── 6. Google Trends (non-fatal; fetcher skips if <7d stale) ─────────────
    try:
        from data.trends_fetcher import fetch_trends_composite
        _run("Google Trends", fetch_trends_composite)
    except ImportError as exc:
        _log(f"  Trends skipped (import error): {exc}")

    # ── 7. Fear & Greed index (non-fatal; Reddit skipped — requires creds) ────
    try:
        from data.sentiment_fetcher import fetch_fear_greed
        _run("Fear & Greed index", fetch_fear_greed)
    except ImportError as exc:
        _log(f"  Fear & Greed skipped (import error): {exc}")

    _log("=========================================")
    _log("Crypto data fetch complete")
    _log("=========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
