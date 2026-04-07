#!/usr/bin/env python3
"""
scripts/run_demand_ic.py — Information Coefficient validation for demand components.

Computes the Spearman Information Coefficient (IC) between each demand index
component and forward BTC returns at 10, 30, and 60-day horizons, on rolling
out-of-sample windows.

What is IC?
-----------
IC = Spearman rank correlation between today's component value and the
return over the next N days.  Range: [-1, 1].

  |IC| < 0.03 : noise — this component has no predictive value
  |IC| 0.03–0.05 : weak signal
  |IC| 0.05–0.10 : moderate signal
  |IC| > 0.10  : strong signal (rare in finance)

A t-statistic of the IC time-series (|t| > 2) confirms the signal is not
due to chance.  This is the minimum bar for enabling any component in the
demand filter.

Why this matters:
-----------------
The demand index currently weights components by config (DEMAND_W_TRENDS = 0.25,
DEMAND_W_MVRV = 0.20, etc.) without empirical validation.  Google Trends is
literature-confirmed as lagging/coincident.  This script measures the actual
predictive power of each component, enabling weight calibration.

Usage:
  python3 scripts/run_demand_ic.py
  python3 scripts/run_demand_ic.py --symbol ETH --horizons 10 30 60
  python3 scripts/run_demand_ic.py --window 180  # min observations per IC estimate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as cfg
from data.fetcher import fetch_ohlcv


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IC validation for demand index components",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol",   default="BTC",       choices=["BTC", "ETH"])
    p.add_argument("--horizons", nargs="+", type=int,  default=[10, 30, 60],
                   help="Forward return horizons in days")
    p.add_argument("--window",   type=int,             default=252,
                   help="Minimum observations for a valid IC estimate")
    p.add_argument("--end",      default=None,
                   help="End date (ISO); defaults to today")
    return p.parse_args()


# ── IC computation ────────────────────────────────────────────────────────────

def compute_ic(
    component: pd.Series,
    forward_returns: pd.Series,
    min_obs: int = 30,
) -> tuple[float, float, int]:
    """
    Compute Spearman IC between component and forward returns.

    Returns (ic, t_stat, n_obs).  NaN if insufficient data.
    """
    common = component.index.intersection(forward_returns.index)
    c = component.loc[common].dropna()
    r = forward_returns.loc[c.index].dropna()
    c = c.loc[r.index]

    n = len(c)
    if n < min_obs:
        return float("nan"), float("nan"), n

    ic, _ = spearmanr(c.values, r.values)
    # t-statistic: IC * sqrt(n-2) / sqrt(1-IC²)
    if abs(ic) >= 1.0:
        t = float("inf")
    else:
        t = ic * np.sqrt(n - 2) / np.sqrt(1.0 - ic ** 2)
    return float(ic), float(t), n


def rolling_ic(
    component: pd.Series,
    forward_returns: pd.Series,
    window: int = 252,
    step: int = 63,         # quarterly steps
    min_obs: int = 60,
) -> pd.DataFrame:
    """
    Compute IC on rolling out-of-sample windows.

    At each step date, IC is computed on the trailing `window` observations
    and recorded.  This tests whether the IC is stable over time or
    concentrated in one regime.

    Returns DataFrame with columns: date, ic, t_stat, n_obs.
    """
    common = component.index.intersection(forward_returns.index)
    c = component.loc[common].dropna()
    r = forward_returns.reindex(c.index).dropna()
    c = c.loc[r.index]

    if len(c) < window:
        return pd.DataFrame(columns=["date", "ic", "t_stat", "n_obs"])

    records = []
    for end_idx in range(window, len(c) + 1, step):
        slice_c = c.iloc[max(0, end_idx - window):end_idx]
        slice_r = r.iloc[max(0, end_idx - window):end_idx]
        date = c.index[end_idx - 1]
        ic, t, n = compute_ic(slice_c, slice_r, min_obs=min_obs)
        records.append({"date": date, "ic": ic, "t_stat": t, "n_obs": n})

    return pd.DataFrame(records)


# ── Format helpers ─────────────────────────────────────────────────────────────

def _signal_label(ic: float, t: float) -> str:
    if np.isnan(ic):
        return "N/A"
    sig = "**" if abs(t) > 2 else ("*" if abs(t) > 1.5 else "")
    if abs(ic) < 0.03:
        return f"{sig}NOISE"
    elif abs(ic) < 0.05:
        return f"{sig}WEAK"
    elif abs(ic) < 0.10:
        return f"{sig}MODERATE"
    else:
        return f"{sig}STRONG"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    symbol = args.symbol

    yf_symbol = cfg.BTC_SYMBOL if symbol == "BTC" else cfg.ETH_SYMBOL
    end_date  = args.end or str(pd.Timestamp.today().date())

    print(f"\n{'═'*70}")
    print(f"  Demand IC Validation — {symbol}  (horizons: {args.horizons}d)")
    print(f"{'═'*70}")

    # ── Fetch price data ────────────────────────────────────────────────────
    print(f"\n  Fetching {yf_symbol} price data...", flush=True)
    df = fetch_ohlcv(yf_symbol, start="2014-01-01", end=end_date)
    print(f"  {len(df)} days  ({df.index[0].date()} → {df.index[-1].date()})")

    # ── Build forward returns for each horizon ──────────────────────────────
    fwd_returns: dict[int, pd.Series] = {}
    for h in args.horizons:
        fwd_returns[h] = (
            df["close"].shift(-h) / df["close"] - 1.0
        ).rename(f"fwd_{h}d")

    # ── Collect demand components ───────────────────────────────────────────
    components: dict[str, pd.Series] = {}

    # Google Trends
    try:
        from data.trends_fetcher import fetch_trends_composite
        print("  Fetching Google Trends...", end=" ", flush=True)
        trends_df = fetch_trends_composite(start="2014-01-01", end=end_date)
        if not trends_df.empty:
            col = "trends_composite" if "trends_composite" in trends_df.columns else trends_df.columns[0]
            components["google_trends"] = trends_df[col]
            print("done")
        else:
            print("empty")
    except Exception as e:
        print(f"skipped ({e})")

    # ETF flows
    try:
        from data.etf_flows_fetcher import fetch_etf_flows
        print("  Fetching ETF flows...", end=" ", flush=True)
        etf_df = fetch_etf_flows(end=end_date)
        if not etf_df.empty:
            if "total_etf_flow_proxy" in etf_df.columns:
                components["etf_flow_proxy"] = etf_df["total_etf_flow_proxy"]
            if "total_etf_volume" in etf_df.columns:
                components["etf_volume"] = etf_df["total_etf_volume"]
            print("done")
    except Exception as e:
        print(f"skipped ({e})")

    # On-chain: MVRV, exchange outflows, active addresses
    try:
        from data.onchain_fetcher import build_onchain_frame
        print("  Fetching on-chain data...", end=" ", flush=True)
        onchain_df = build_onchain_frame(symbol, end=end_date)
        for col in ["mvrv_zscore", "exchange_outflows", "active_addresses"]:
            if col in onchain_df.columns:
                components[col] = onchain_df[col]
        print("done")
    except Exception as e:
        print(f"skipped ({e})")

    # Fear & Greed Index
    try:
        from data.sentiment_fetcher import fetch_fear_greed
        print("  Fetching Fear & Greed...", end=" ", flush=True)
        fg_df = fetch_fear_greed(end=end_date)
        if not fg_df.empty:
            col = "fear_greed" if "fear_greed" in fg_df.columns else fg_df.columns[0]
            components["fear_greed"] = fg_df[col]
            print("done")
    except Exception as e:
        print(f"skipped ({e})")

    # Spot volume (from main price data)
    components["spot_volume"] = df["volume"].rolling(7).mean()

    if not components:
        print("\n  No demand components available.  Install data dependencies and retry.")
        return

    print(f"\n  Components available: {list(components.keys())}")

    # ── Compute IC per component per horizon ────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  FULL-SAMPLE IC  (all available data)")
    print(f"{'─'*70}")

    w_comp = 22
    horizon_header = "".join(f"  {'IC/t/sig':>22}" for _ in args.horizons)
    col_headers    = "".join(f"  {f'{h}d IC':>8} {'t':>6} {'signal':>7}" for h in args.horizons)
    print(f"  {'Component':<{w_comp}}" + col_headers)
    print(f"  {'─'*w_comp}" + "".join(f"  {'─'*8} {'─'*6} {'─'*7}" for _ in args.horizons))

    ic_summary: dict[str, dict] = {}

    for comp_name, comp_series in components.items():
        row = f"  {comp_name:<{w_comp}}"
        ic_summary[comp_name] = {}
        for h in args.horizons:
            ic, t, n = compute_ic(comp_series, fwd_returns[h], min_obs=args.window // 4)
            sig = _signal_label(ic, t)
            ic_str = f"{ic:+.3f}" if not np.isnan(ic) else "  N/A"
            t_str  = f"{t:+.2f}"  if not np.isnan(t)  else "  N/A"
            row += f"  {ic_str:>8} {t_str:>6} {sig:>7}"
            ic_summary[comp_name][h] = {"ic": ic, "t": t, "n": n}
        print(row)

    print(f"{'─'*70}")
    print(f"  ** |t| > 2 (statistically significant at 95%)  * |t| > 1.5")
    print(f"  NOISE = |IC| < 0.03   WEAK = 0.03–0.05   MODERATE = 0.05–0.10   STRONG > 0.10")

    # ── Recommended weights ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  WEIGHT RECOMMENDATIONS  (based on 30d IC)")
    print(f"{'─'*70}")
    primary_horizon = 30 if 30 in args.horizons else args.horizons[0]
    ic_vals: dict[str, float] = {}
    for comp_name in components:
        ic_vals[comp_name] = abs(ic_summary[comp_name].get(primary_horizon, {}).get("ic", 0.0) or 0.0)

    total_ic = sum(ic_vals.values()) or 1.0
    print(f"  {'Component':<{w_comp}} {'|IC@30d|':>10} {'Rec. weight':>12} {'Current weight':>15}")
    print(f"  {'─'*w_comp} {'─'*10} {'─'*12} {'─'*15}")

    weight_map = {
        "google_trends":    getattr(cfg, "DEMAND_W_TRENDS",   0.25),
        "etf_flow_proxy":   getattr(cfg, "DEMAND_W_ETF",      0.20),
        "etf_volume":       getattr(cfg, "DEMAND_W_ETF",      0.20),
        "mvrv_zscore":      getattr(cfg, "DEMAND_W_MVRV",     0.20),
        "exchange_outflows":getattr(cfg, "DEMAND_W_OUTFLOWS",  0.10),
        "active_addresses": 0.05,
        "fear_greed":       0.10,
        "spot_volume":      getattr(cfg, "DEMAND_W_VOLUME",   0.25),
    }
    for comp_name in components:
        ic_abs   = ic_vals[comp_name]
        rec_w    = ic_abs / total_ic if total_ic > 0 else 0.0
        cur_w    = weight_map.get(comp_name, 0.0)
        flag = " ← REMOVE (noise)" if ic_abs < 0.03 else ""
        print(f"  {comp_name:<{w_comp}} {ic_abs:>10.4f} {rec_w:>11.1%} {cur_w:>14.1%}{flag}")

    print(f"{'─'*70}")
    print(f"  Recommendation: disable USE_DEMAND_FILTER for components with |IC| < 0.03.")
    print(f"  Calibrate DEMAND_W_* to the 'Rec. weight' column above.")

    # ── Rolling IC stability ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  ROLLING IC STABILITY  (quarterly windows, {primary_horizon}d horizon)")
    print(f"{'─'*70}")
    print(f"  {'Component':<{w_comp}} {'Mean IC':>9} {'Std IC':>8} {'Min IC':>8} {'Max IC':>8} {'Stable?':>9}")
    print(f"  {'─'*w_comp} {'─'*9} {'─'*8} {'─'*8} {'─'*8} {'─'*9}")

    for comp_name, comp_series in components.items():
        roll = rolling_ic(comp_series, fwd_returns[primary_horizon],
                          window=args.window, step=63, min_obs=30)
        if roll.empty or roll["ic"].dropna().empty:
            print(f"  {comp_name:<{w_comp}} {'N/A':>9}")
            continue
        ic_series = roll["ic"].dropna()
        mean_ic = ic_series.mean()
        std_ic  = ic_series.std()
        min_ic  = ic_series.min()
        max_ic  = ic_series.max()
        stable  = "YES" if std_ic < 0.05 else ("MODERATE" if std_ic < 0.10 else "NO")
        print(f"  {comp_name:<{w_comp}} {mean_ic:>+9.3f} {std_ic:>8.3f} "
              f"{min_ic:>+8.3f} {max_ic:>+8.3f} {stable:>9}")

    print(f"{'─'*70}")
    print(f"  Stable IC = consistent signal across market regimes.")
    print(f"  Unstable IC (std > 0.10) = regime-dependent; risky to include in filter.")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
