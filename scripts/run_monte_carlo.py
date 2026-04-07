#!/usr/bin/env python3
"""
scripts/run_monte_carlo.py — Monte Carlo robustness analysis.

Runs two analyses:
  1. Trade resampling  — shuffles closed trades 1,000+ times, plots outcome distribution
  2. Parameter sensitivity — varies fees, slippage, Z thresholds across a grid

Usage:
  python3 scripts/run_monte_carlo.py --symbol BTC
  python3 scripts/run_monte_carlo.py --symbol BTC --sims 2000 --chart
  python3 scripts/run_monte_carlo.py --symbol BTC --sensitivity
  python3 scripts/run_monte_carlo.py --symbol BTC --walk-forward --regime
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
from backtest.engine import build_features, build_features_walk_forward, run_backtest, buy_and_hold
from backtest.metrics import cagr, max_drawdown, sharpe_ratio
from backtest.monte_carlo import run_trade_resample, run_param_sensitivity, plot_monte_carlo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo robustness analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol",       default="BTC", choices=["BTC", "ETH"])
    p.add_argument("--start",        default=cfg.START_DATE)
    p.add_argument("--end",          default=str(date.today()))
    p.add_argument("--sims",         type=int,   default=1_000,
                   help="Number of Monte Carlo simulations")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--chart",        action="store_true",
                   help="Save histogram and fan charts to reports/charts/")
    p.add_argument("--sensitivity",  action="store_true",
                   help="Also run parameter sensitivity analysis")
    p.add_argument("--walk-forward", action="store_true")
    p.add_argument("--regime",       action="store_true")
    p.add_argument("--no-trend",     action="store_true")
    p.add_argument("--no-volume",    action="store_true")
    p.add_argument("--no-macro",     action="store_true")
    p.add_argument("--buy-z",        type=float, default=cfg.BUY_THRESHOLD)
    p.add_argument("--sell-z",       type=float, default=cfg.SELL_THRESHOLD)
    p.add_argument("--out",          default=None,
                   help="Markdown report output path")
    return p.parse_args()


def main() -> None:
    import types
    args = parse_args()

    symbol    = args.symbol
    yf_symbol = cfg.BTC_SYMBOL if symbol == "BTC" else cfg.ETH_SYMBOL
    genesis   = cfg.BTC_GENESIS if symbol == "BTC" else cfg.ETH_GENESIS

    # Build a config override
    run_cfg = types.ModuleType("cfg_mc")
    run_cfg.__dict__.update({k: v for k, v in vars(cfg).items()
                             if not k.startswith("__")})
    run_cfg.BUY_THRESHOLD     = args.buy_z
    run_cfg.SELL_THRESHOLD    = args.sell_z
    run_cfg.USE_TREND_FILTER  = not args.no_trend
    run_cfg.USE_VOLUME_FILTER = not args.no_volume
    run_cfg.USE_MACRO_FILTER  = not args.no_macro

    print(f"\n{'═'*60}")
    print(f"  {symbol} Monte Carlo — {args.sims:,} simulations")
    print(f"{'═'*60}")

    print(f"\n  Fetching {yf_symbol} data...", flush=True)
    df = fetch_ohlcv(yf_symbol, start="2014-01-01", end=args.end)
    print(f"  {len(df)} days  ({df.index[0].date()} → {df.index[-1].date()})")

    if args.walk_forward:
        print(f"  Building walk-forward features...", flush=True)
        features = build_features_walk_forward(df, genesis_date=genesis, cfg=run_cfg)
    else:
        print(f"  Building features...", flush=True)
        features = build_features(df, genesis_date=genesis, cfg=run_cfg,
                                  use_regime=args.regime)

    print(f"  Running base backtest...", flush=True)
    equity, trades, _open_trade = run_backtest(features, cfg=run_cfg, start_date=args.start)
    bah            = buy_and_hold(features, cfg=run_cfg, start_date=args.start)
    bah            = bah.reindex(equity.index).ffill()
    bah            = bah / bah.iloc[0] * run_cfg.INITIAL_CAPITAL

    print(f"\n  Base backtest results:")
    print(f"    CAGR:        {cagr(equity)*100:+.1f}%  (B&H: {cagr(bah)*100:+.1f}%)")
    print(f"    Sharpe:      {sharpe_ratio(equity):.2f}  (B&H: {sharpe_ratio(bah):.2f})")
    print(f"    Max DD:      {max_drawdown(equity)*100:.1f}%  (B&H: {max_drawdown(bah)*100:.1f}%)")
    print(f"    Trades:      {len(trades)}")

    if trades.empty or "pnl_pct" not in trades.columns:
        print("\n  No trades found — cannot run Monte Carlo. Try a wider date range.")
        return

    # ── 1. Trade resampling ──────────────────────────────────────────────────
    print(f"\n  Running {args.sims:,} trade-resample simulations...", flush=True)
    mc = run_trade_resample(
        trades,
        initial_capital=run_cfg.INITIAL_CAPITAL,
        n_sims=args.sims,
        seed=args.seed,
    )

    bah_total_return = float(bah.iloc[-1] / run_cfg.INITIAL_CAPITAL - 1.0)
    beat_prob = (mc["sim_results"]["total_return"] > bah_total_return).mean() * 100

    print(f"\n  {'─'*56}")
    print(f"  Monte Carlo Percentile Summary ({mc['n_trades']} trades resampled)")
    print(f"  {'─'*56}")
    print(mc["summary"])
    print(f"\n  P(beat buy-and-hold): {beat_prob:.1f}%")
    print(f"  B&H total return:     {bah_total_return*100:+.1f}%")

    # ── 2. Charts ────────────────────────────────────────────────────────────
    if args.chart:
        print(f"\n  Generating charts...", flush=True)
        charts_dir = ROOT / "reports" / "charts"
        saved = plot_monte_carlo(mc, bah_total_return=bah_total_return,
                                 out_dir=charts_dir, symbol=symbol)
        for p in saved:
            print(f"    Saved → {p}")

    # ── 3. Parameter sensitivity ─────────────────────────────────────────────
    sensitivity_df = pd.DataFrame()
    if args.sensitivity:
        print(f"\n  Running parameter sensitivity (fees × slippage × buy-Z grid)...",
              flush=True)
        sensitivity_df = run_param_sensitivity(
            features, cfg=run_cfg, start_date=args.start,
        )
        if not sensitivity_df.empty:
            print(f"\n  Top 10 configs by Sharpe:")
            print(f"  {'Label':<40} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'Trades':>7}")
            print(f"  {'─'*40} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
            for _, row in sensitivity_df.head(10).iterrows():
                print(f"  {row['label']:<40} "
                      f"{row['sharpe']:>7.2f} "
                      f"{row['cagr']*100:>6.1f}% "
                      f"{row['max_dd']*100:>6.1f}% "
                      f"{row['n_trades']:>7.0f}")

    # ── 4. Save report ────────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else (
        ROOT / "reports" / f"monte_carlo_{symbol}_{date.today()}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# {symbol} Monte Carlo Analysis — {date.today()}")
    lines.append(f"\n**Simulations:** {args.sims:,}  ·  **Trades:** {mc['n_trades']}  ·  "
                 f"**Seed:** {args.seed}")
    lines.append(f"\n## Percentile Summary\n\n```\n{mc['summary']}\n```")
    lines.append(f"\n**P(beat buy-and-hold): {beat_prob:.1f}%**  "
                 f"(B&H total return: {bah_total_return*100:+.1f}%)")
    if args.chart:
        lines.append(f"\n## Charts\n")
        for p in saved:
            lines.append(f"![{p.stem}](charts/{p.name})")
    if not sensitivity_df.empty:
        lines.append(f"\n## Parameter Sensitivity (top 15 by Sharpe)\n")
        lines.append(sensitivity_df.head(15).to_markdown(index=False, floatfmt=".3f"))
    lines.append(f"\n## Base Backtest\n")
    lines.append(f"- CAGR: {cagr(equity)*100:+.1f}%  (B&H: {cagr(bah)*100:+.1f}%)")
    lines.append(f"- Sharpe: {sharpe_ratio(equity):.2f}  (B&H: {sharpe_ratio(bah):.2f})")
    lines.append(f"- Max DD: {max_drawdown(equity)*100:.1f}%")

    out_path.write_text("\n".join(lines))
    print(f"\n  Report saved → {out_path}")


if __name__ == "__main__":
    main()
