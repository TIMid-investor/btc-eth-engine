#!/usr/bin/env python3
"""
scripts/run_signals.py — Live signal dashboard for BTC and ETH.

Shows the current Z-score, power-law expected price, deviation from the
growth curve, and filter states for both assets.

Usage:
  python3 scripts/run_signals.py
  python3 scripts/run_signals.py --symbol BTC
  python3 scripts/run_signals.py --days 30     # show last N rows
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config as cfg
from data.fetcher import fetch_ohlcv
from backtest.engine import build_features
from models.power_law import fit_power_law, print_fit_summary, _days_since


# ── Signal interpretation ──────────────────────────────────────────────────────

def _signal_label(z: float, buy_thresh: float, sell_thresh: float) -> str:
    if np.isnan(z):
        return "NO SIGNAL (burn-in)"
    if z < -3.0:
        return "STRONG BUY  ▲▲▲"
    if z < -buy_thresh:
        return f"BUY         ▲▲  (Z={z:.2f})"
    if z > 3.0:
        return "STRONG SELL ▼▼▼"
    if z > sell_thresh:
        return f"SELL        ▼▼  (Z={z:.2f})"
    if z < -0.5:
        return f"Mildly oversold (Z={z:.2f})"
    if z > 0.5:
        return f"Mildly overbought (Z={z:.2f})"
    return f"Neutral (Z={z:.2f})"


def _bar(z: float, width: int = 20) -> str:
    """ASCII bar chart centred at 0."""
    half = width // 2
    if np.isnan(z):
        return " " * width + "|" + " " * width
    clipped = max(-3.0, min(3.0, z))
    filled  = int(abs(clipped) / 3.0 * half)
    if clipped >= 0:
        left  = " " * half
        right = "█" * filled + " " * (half - filled)
    else:
        left  = " " * (half - filled) + "█" * filled
        right = " " * half
    return left + "|" + right


def _filter_icon(val: bool | int | None) -> str:
    if val is None:
        return "?"
    if isinstance(val, bool) or val in (0, 1, -1):
        if val is True or val == 1:
            return "✓"
        if val is False or val == 0:
            return "✗"
    return str(val)


# ── Per-symbol dashboard ───────────────────────────────────────────────────────

def print_dashboard(symbol: str, n_rows: int = 1) -> None:
    yf_symbol = cfg.BTC_SYMBOL if symbol == "BTC" else cfg.ETH_SYMBOL
    genesis   = cfg.BTC_GENESIS if symbol == "BTC" else cfg.ETH_GENESIS

    print(f"\n  Fetching {yf_symbol}...", end=" ", flush=True)
    df = fetch_ohlcv(yf_symbol, start="2014-01-01")
    print(f"{len(df)} days", flush=True)

    features = build_features(df, genesis_date=genesis, cfg=cfg)

    # Most recent row with valid Z-score
    valid = features.dropna(subset=["zscore"])
    if valid.empty:
        print(f"  [{symbol}] No valid Z-score rows — need more history.")
        return

    tail = valid.tail(n_rows)
    latest = valid.iloc[-1]

    z           = float(latest["zscore"])
    price       = float(latest["close"])
    expected    = float(latest["expected_price"])
    log_dev     = float(latest["log_deviation"])
    trend       = int(latest["trend"])
    volume_ok   = bool(latest["volume_ok"])
    macro_ok    = bool(latest["macro_ok"])
    target_pos  = float(latest["target_position"])
    as_of       = latest.name.date()

    print(f"\n{'═'*64}")
    print(f"  {symbol}  ·  as of {as_of}")
    print(f"{'═'*64}")
    print(f"  Price:          ${price:>12,.2f}")
    print(f"  Expected:       ${expected:>12,.2f}")
    print(f"  Deviation:      {log_dev:>+.4f}  ({(np.exp(log_dev)-1)*100:>+.1f}%)")
    print(f"")
    print(f"  Z-score:  {z:>+7.3f}  {_bar(z)}")
    print(f"  Signal:   {_signal_label(z, cfg.BUY_THRESHOLD, cfg.SELL_THRESHOLD)}")
    print(f"  Target:   {target_pos:>+.2f} ({'LONG' if target_pos > 0 else 'SHORT' if target_pos < 0 else 'FLAT'})")
    print(f"")
    print(f"  Filters:")
    trend_str = "UP ✓" if trend > 0 else "DOWN — blocking longs" if trend < 0 else "FLAT"
    print(f"    Trend (20-wk EMA slope): {trend_str}"
          + ("" if cfg.USE_TREND_FILTER else "  [disabled]"))
    print(f"    Volume OK:               {'Yes ✓' if volume_ok else 'No — low volume'}"
          + ("" if cfg.USE_VOLUME_FILTER else "  [disabled]"))
    print(f"    Macro OK:                {'Yes ✓' if macro_ok else 'No — shock drawdown'}"
          + ("" if cfg.USE_MACRO_FILTER else "  [disabled]"))

    # Curve fit summary
    if cfg.CURVE_MODEL == "power_law":
        params = fit_power_law(features["close"], genesis)
        print(f"")
        print(f"  Power-law curve:")
        print(f"    price = {params['a']:.4e} × days^{params['b']:.4f}  (R²={params['r_squared']:.4f})")

    # Show last N rows if requested
    if n_rows > 1:
        print(f"\n  Last {n_rows} days:")
        print(f"  {'Date':>12}  {'Close':>10}  {'Expected':>10}  {'Z-score':>8}  {'Target':>7}")
        print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*7}")
        for dt, row in tail.iterrows():
            _z = row["zscore"]
            _t = row["target_position"]
            print(f"  {str(dt.date()):>12}  {row['close']:>10,.0f}  "
                  f"{row['expected_price']:>10,.0f}  "
                  f"{_z:>+8.3f}  "
                  f"{_t:>+6.2f} {'L' if _t > 0 else 'S' if _t < 0 else '-'}")

    print(f"{'═'*64}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live BTC/ETH signal dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default=None, choices=["BTC", "ETH"],
                   help="Show only this asset (default: both)")
    p.add_argument("--days",   type=int, default=1,
                   help="Show last N days of signals in the table")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [args.symbol] if args.symbol else ["BTC", "ETH"]
    print(f"\n  Crypto Signal Dashboard — {date.today()}")
    for sym in symbols:
        print_dashboard(sym, n_rows=args.days)
    print()


if __name__ == "__main__":
    main()
