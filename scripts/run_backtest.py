#!/usr/bin/env python3
"""
scripts/run_backtest.py — Run the BTC or ETH swing trading backtest.

Usage:
  python3 scripts/run_backtest.py                        # BTC, default dates
  python3 scripts/run_backtest.py --symbol ETH
  python3 scripts/run_backtest.py --symbol BTC --start 2018-01-01
  python3 scripts/run_backtest.py --symbol BTC --curve log_ema
  python3 scripts/run_backtest.py --symbol BTC --no-trend --no-volume
  python3 scripts/run_backtest.py --symbol BTC --long-short
  python3 scripts/run_backtest.py --symbol BTC --buy-z 2.0 --sell-z 2.0
  python3 scripts/run_backtest.py --symbol BTC --fees 0.002 --slippage 0.001
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from datetime import date
from pathlib import Path

# Project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config as cfg
from data.fetcher import fetch_ohlcv
from backtest.engine import build_features, build_features_walk_forward, run_backtest, buy_and_hold
from models.ml_overlay import run_backtest_with_ml
from backtest.metrics import summary_table, cagr, max_drawdown, sharpe_ratio
from models.power_law import print_fit_summary, fit_power_law


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BTC / ETH power-law swing trading backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol",     default="BTC",      choices=["BTC", "ETH"],
                   help="Asset to backtest")
    p.add_argument("--start",      default=cfg.START_DATE,
                   help="Backtest start date (ISO)")
    p.add_argument("--end",        default=str(date.today()),
                   help="Backtest end date (ISO)")
    p.add_argument("--curve",      default=cfg.CURVE_MODEL,
                   choices=["power_law", "log_ema"],
                   help="Expected-price model")
    p.add_argument("--buy-z",      type=float, default=cfg.BUY_THRESHOLD,
                   help="Z-score threshold to enter long (|Z| must exceed this)")
    p.add_argument("--sell-z",     type=float, default=cfg.SELL_THRESHOLD,
                   help="Z-score threshold to exit / go short")
    p.add_argument("--exit-z",     type=float, default=cfg.EXIT_THRESHOLD,
                   help="Z-score band around zero where position is closed")
    p.add_argument("--z-window",   type=int,   default=cfg.ZSCORE_WINDOW,
                   help="Rolling Z-score window in days")
    p.add_argument("--fees",       type=float, default=cfg.FEE_RATE,
                   help="Fee rate per trade (one-way fraction)")
    p.add_argument("--slippage",   type=float, default=cfg.SLIPPAGE,
                   help="Slippage per trade (one-way fraction)")
    p.add_argument("--capital",    type=float, default=cfg.INITIAL_CAPITAL,
                   help="Starting capital ($)")
    p.add_argument("--long-short", action="store_true",
                   help="Allow short positions (default: long-only)")
    p.add_argument("--no-trend",   action="store_true",
                   help="Disable trend filter")
    p.add_argument("--no-volume",  action="store_true",
                   help="Disable volume filter")
    p.add_argument("--no-macro",      action="store_true",
                   help="Disable macro shock filter")
    p.add_argument("--walk-forward",  action="store_true",
                   help="Use walk-forward power-law refitting (no look-ahead in curve params)")
    p.add_argument("--refit-months",  type=int, default=3,
                   help="Months between curve refits in walk-forward mode")
    p.add_argument("--regime",        action="store_true",
                   help="Enable regime-aware Z-score thresholds (halving cycle + bull/bear)")
    p.add_argument("--ml",            action="store_true",
                   help="Enable ML overlay (logistic classifier gates trade entries)")
    p.add_argument("--ml-threshold",  type=float, default=0.55,
                   help="Minimum ML confidence to enter a trade (0-1)")
    p.add_argument("--hold-through-cycle", action="store_true",
                   help="Buy on deep Z signal, hold until overbought (Z > sell_thresh) rather than exiting on mean-reversion")
    p.add_argument("--out",           default=None,
                   help="Path to write markdown report (default: reports/backtest_<SYMBOL>_<DATE>.md)")
    return p.parse_args()


def _override_cfg(args: argparse.Namespace) -> types.ModuleType:
    """Return a shallow copy of cfg with CLI overrides applied."""
    overrides = types.ModuleType("cfg_override")
    overrides.__dict__.update({k: v for k, v in vars(cfg).items()
                                if not k.startswith("__")})
    overrides.CURVE_MODEL          = args.curve
    overrides.BUY_THRESHOLD        = args.buy_z
    overrides.SELL_THRESHOLD       = args.sell_z
    overrides.EXIT_THRESHOLD       = args.exit_z
    overrides.ZSCORE_WINDOW        = args.z_window
    overrides.FEE_RATE             = args.fees
    overrides.SLIPPAGE             = args.slippage
    overrides.INITIAL_CAPITAL      = args.capital
    overrides.LONG_ONLY            = not args.long_short
    overrides.HOLD_THROUGH_CYCLE   = args.hold_through_cycle
    overrides.USE_TREND_FILTER     = not args.no_trend
    overrides.USE_VOLUME_FILTER    = not args.no_volume
    overrides.USE_MACRO_FILTER     = not args.no_macro
    return overrides


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(
    symbol: str,
    args: argparse.Namespace,
    run_cfg,
    features: pd.DataFrame,
    equity: pd.Series,
    trades: pd.DataFrame,
    bah: pd.Series,
) -> str:
    lines: list[str] = []
    A = lines.append

    A(f"# {symbol} Swing Trading Backtest — {date.today()}")
    A("")
    A(f"**Period:** {equity.index[0].date()} → {equity.index[-1].date()}")
    A(f"**Curve model:** {run_cfg.CURVE_MODEL}  ·  "
      f"**Z-score window:** {run_cfg.ZSCORE_WINDOW}d  ·  "
      f"**Buy threshold:** {run_cfg.BUY_THRESHOLD}  ·  "
      f"**Sell threshold:** {run_cfg.SELL_THRESHOLD}")
    hold_mode = getattr(run_cfg, "HOLD_THROUGH_CYCLE", False)
    A(f"**Filters:** trend={'ON' if run_cfg.USE_TREND_FILTER else 'OFF'}  "
      f"volume={'ON' if run_cfg.USE_VOLUME_FILTER else 'OFF'}  "
      f"macro={'ON' if run_cfg.USE_MACRO_FILTER else 'OFF'}  ·  "
      f"**Mode:** {'Long/Short' if not run_cfg.LONG_ONLY else 'Long-Only'}"
      + ("  ·  **Hold-Through-Cycle: ON**" if hold_mode else ""))
    A(f"**Fees:** {run_cfg.FEE_RATE*100:.2f}%  ·  "
      f"**Slippage:** {run_cfg.SLIPPAGE*100:.2f}%  ·  "
      f"**Starting capital:** ${run_cfg.INITIAL_CAPITAL:,.0f}")
    A("")

    A("## Performance Summary")
    A("")
    A("```")
    A(summary_table(equity, bah, trades,
                    label=f"{symbol} Strategy",
                    bench_label="Buy & Hold"))
    A("```")
    A("")

    A("## Trade Log")
    A("")
    if trades.empty:
        A("*No completed trades in this period.*")
    else:
        A(f"  {'#':<4} {'Entry':>12} {'Exit':>12} {'Dir':>6} "
          f"{'Entry $':>10} {'Exit $':>10} {'Z entry':>8} {'Z exit':>7} {'P&L':>10} {'P&L %':>7}")
        A(f"  {'─'*4} {'─'*12} {'─'*12} {'─'*6} "
          f"{'─'*10} {'─'*10} {'─'*8} {'─'*7} {'─'*10} {'─'*7}")
        for i, row in trades.iterrows():
            A(f"  {i+1:<4} {str(row.get('entry_date',''))[:10]:>12} "
              f"{str(row.get('exit_date',''))[:10]:>12} "
              f"{str(row.get('direction','')):>6} "
              f"{row.get('entry_price', float('nan')):>10,.0f} "
              f"{row.get('exit_price',  float('nan')):>10,.0f} "
              f"{row.get('entry_z',     float('nan')):>8.2f} "
              f"{row.get('exit_z',      float('nan')):>7.2f} "
              f"{row.get('pnl',         float('nan')):>+10,.0f} "
              f"{row.get('pnl_pct',     float('nan'))*100:>+6.1f}%")
    A("")

    # Regime summary (by Z-score quartile at entry)
    if not trades.empty and "entry_z" in trades.columns:
        A("## Entry Z-Score Distribution")
        A("")
        A("```")
        A(f"  {'Z range':<14} {'Trades':>7} {'Win %':>7} {'Avg P&L':>10}")
        A(f"  {'─'*14} {'─'*7} {'─'*7} {'─'*10}")
        bins = [(-10, -3), (-3, -2), (-2, -1.5), (-1.5, -1), (-1, 0), (0, 10)]
        for lo, hi in bins:
            sub = trades[(trades["entry_z"] >= lo) & (trades["entry_z"] < hi)]
            if sub.empty:
                continue
            wr  = (sub["pnl"] > 0).mean() * 100 if "pnl" in sub else float("nan")
            avg = sub["pnl"].mean() if "pnl" in sub else float("nan")
            A(f"  [{lo:+.1f}, {hi:+.1f})  {len(sub):>7} {wr:>6.0f}% {avg:>+10,.0f}")
        A("```")
        A("")

    A("## Curve Fit Details")
    A("")
    if run_cfg.CURVE_MODEL == "power_law":
        from models.power_law import fit_power_law as _fp, _days_since
        import numpy as np
        genesis = cfg.BTC_GENESIS if symbol == "BTC" else cfg.ETH_GENESIS
        params = _fp(features["close"], genesis)
        today_t = _days_since(pd.DatetimeIndex([pd.Timestamp.today()]), genesis)[0]
        today_ex = params["a"] * today_t ** params["b"]
        today_ac = float(features["close"].iloc[-1])
        A(f"  price = {params['a']:.4e} × days_since_genesis ^ {params['b']:.4f}")
        A(f"  R² (log scale): {params['r_squared']:.4f}")
        A(f"  Today:  actual ${today_ac:>12,.0f}   expected ${today_ex:>12,.0f}"
          f"   ratio {today_ac/today_ex:.3f}")
    else:
        A(f"  Log-EMA span: {run_cfg.LOG_EMA_SPAN} days")
    A("")

    A("## Caveats")
    A("")
    wf = getattr(args, "walk_forward", False)
    if wf:
        A("- **Walk-forward mode**: power-law refit every quarter using only past data. No look-ahead bias in curve.")
    else:
        A("- Power-law exponent is fit on all available data (look-ahead in the curve itself). Use `--walk-forward` to eliminate this.")
    A("- Z-score uses a rolling trailing window — no look-ahead in signal generation.")
    A("- No tax, funding costs, or borrow fees for shorts modelled.")
    A("- Past performance of a mean-reversion model in a trending asset is not indicative of future results.")
    A("")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    run_cfg = _override_cfg(args)

    symbol    = args.symbol
    yf_symbol = cfg.BTC_SYMBOL if symbol == "BTC" else cfg.ETH_SYMBOL
    genesis   = cfg.BTC_GENESIS if symbol == "BTC" else cfg.ETH_GENESIS

    walk_forward = args.walk_forward

    print(f"\n{'═'*60}")
    print(f"  {symbol} Swing Trading Backtest"
          + (" [WALK-FORWARD]" if walk_forward else ""))
    print(f"{'═'*60}")
    print(f"  Curve model : {run_cfg.CURVE_MODEL}"
          + (f"  (refit every {args.refit_months}mo)" if walk_forward else " (full-history fit)"))
    print(f"  Z thresholds: buy < -{run_cfg.BUY_THRESHOLD}  ·  sell > {run_cfg.SELL_THRESHOLD}")
    print(f"  Filters     : trend={run_cfg.USE_TREND_FILTER}  volume={run_cfg.USE_VOLUME_FILTER}  macro={run_cfg.USE_MACRO_FILTER}")
    print(f"  Long-only   : {run_cfg.LONG_ONLY}")
    print(f"{'─'*60}")

    print(f"\n  Fetching {yf_symbol} data...", flush=True)
    df = fetch_ohlcv(yf_symbol, start="2014-01-01", end=args.end)
    print(f"  {len(df)} trading days  ({df.index[0].date()} → {df.index[-1].date()})")

    use_regime = args.regime

    if walk_forward:
        print(f"  Building walk-forward features (quarterly refits)...", flush=True)
        features = build_features_walk_forward(
            df, genesis_date=genesis, cfg=run_cfg,
            refit_months=args.refit_months,
        )
        print(f"  {features['curve_fit_date'].nunique()} unique curve fits applied")
        if use_regime:
            # Apply regime layer on top of walk-forward features
            from models.regime import build_regime_frame, apply_regime_to_target
            include_halving = (genesis == cfg.BTC_GENESIS)
            regime_df = build_regime_frame(features["close"], run_cfg.BUY_THRESHOLD, include_halving)
            features["halving_phase"] = regime_df["halving_phase"]
            features["price_regime"]  = regime_df["price_regime"]
            features["regime"]        = regime_df["regime"]
            features["threshold_mult"] = regime_df["threshold_mult"]
            features["target_position"] = apply_regime_to_target(
                features["target_position"], features["zscore"],
                regime_df, run_cfg.BUY_THRESHOLD,
            )
    else:
        print(f"  Building features ({run_cfg.CURVE_MODEL} curve"
              + (", regime-aware" if use_regime else "") + ")...", flush=True)
        features = build_features(df, genesis_date=genesis, cfg=run_cfg,
                                  use_regime=use_regime)
        if run_cfg.CURVE_MODEL == "power_law" and not walk_forward:
            from models.power_law import fit_power_law as _fp
            params = _fp(features["close"], genesis)
            print_fit_summary(params, features["close"])

    if use_regime and "regime" in features.columns:
        print(f"  Regime distribution:")
        counts = features["regime"].value_counts()
        for regime, n in counts.head(6).items():
            print(f"    {regime:<30} {n:>5} days")

    print(f"\n  Running backtest from {args.start}"
          + (" [ML overlay]" if args.ml else "") + "...", flush=True)
    if args.ml:
        equity, trades, conf_log = run_backtest_with_ml(
            features, cfg=run_cfg, start_date=args.start,
            confidence_threshold=args.ml_threshold,
        )
    else:
        equity, trades = run_backtest(features, cfg=run_cfg, start_date=args.start)
    bah            = buy_and_hold(features, cfg=run_cfg, start_date=args.start)

    # Align buy-and-hold to strategy dates
    bah = bah.reindex(equity.index).ffill()
    # Rescale bah so both start at INITIAL_CAPITAL
    bah = bah / bah.iloc[0] * run_cfg.INITIAL_CAPITAL

    print(f"\n{'─'*60}")
    print(summary_table(equity, bah, trades,
                        label=f"{symbol} Strategy",
                        bench_label="Buy & Hold"))

    print(f"\n  Closed trades: {len(trades)}")

    # ── Save report ──────────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else (
        ROOT / "reports" / f"backtest_{symbol}_{date.today()}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(symbol, args, run_cfg, features, equity, trades, bah)
    out_path.write_text(report)
    print(f"\n  Report saved → {out_path}")


if __name__ == "__main__":
    main()
