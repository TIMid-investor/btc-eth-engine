#!/usr/bin/env python3
"""
scripts/param_sweep.py — Grid search over key model parameters.

Sweeps buy_z, sell_z, z_window, and filter combinations.
For each combination: runs full backtest, records CAGR / Sharpe / MaxDD.
Outputs ranked table + Pareto-optimal configs + overfitting flags.
Optionally plots the Sharpe vs MaxDD efficient frontier.

Usage:
  python3 scripts/param_sweep.py                     # BTC, default grid
  python3 scripts/param_sweep.py --symbol ETH
  python3 scripts/param_sweep.py --symbol BTC --chart
  python3 scripts/param_sweep.py --symbol BTC --top 20
  python3 scripts/param_sweep.py --symbol BTC --quick    # smaller grid, faster
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import types
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config as cfg
from data.fetcher import fetch_ohlcv
from backtest.engine import build_features, run_backtest, buy_and_hold
from backtest.metrics import cagr, sharpe_ratio, max_drawdown, calmar_ratio


# ── Parameter grid ────────────────────────────────────────────────────────────

FULL_GRID = {
    "buy_z":    [1.0, 1.5, 2.0, 2.5, 3.0],
    "sell_z":   [1.0, 1.5, 2.0, 2.5],
    "z_window": [180, 365, 545, 730],
    "trend":    [True, False],
    "volume":   [True, False],
    "macro":    [True, False],
}

QUICK_GRID = {
    "buy_z":    [1.5, 2.0, 2.5],
    "sell_z":   [1.5, 2.0],
    "z_window": [365, 730],
    "trend":    [True, False],
    "volume":   [True],
    "macro":    [True],
}


# ── Config override factory ────────────────────────────────────────────────────

def _make_cfg(
    buy_z: float,
    sell_z: float,
    z_window: int,
    trend: bool,
    volume: bool,
    macro: bool,
) -> types.ModuleType:
    oc = types.ModuleType("sweep_cfg")
    oc.__dict__.update({k: v for k, v in vars(cfg).items() if not k.startswith("__")})
    oc.BUY_THRESHOLD     = buy_z
    oc.SELL_THRESHOLD    = sell_z
    oc.EXIT_THRESHOLD    = min(buy_z * 0.20, 0.30)  # exit at 20% of entry threshold
    oc.ZSCORE_WINDOW     = z_window
    oc.ZSCORE_MIN_PERIODS = max(90, z_window // 2)
    oc.USE_TREND_FILTER  = trend
    oc.USE_VOLUME_FILTER = volume
    oc.USE_MACRO_FILTER  = macro
    oc.LONG_ONLY         = True
    return oc


# ── Single run ────────────────────────────────────────────────────────────────

def _run_one(
    features_cache: dict,
    genesis: str,
    run_cfg,
    start_date: str,
) -> dict | None:
    """
    Run backtest for one parameter combo.
    features_cache maps z_window → pre-built base features (close/volume only).
    We rebuild the full feature set per config to get the right Z-score window.
    """
    key = run_cfg.ZSCORE_WINDOW
    if key not in features_cache:
        return None

    raw = features_cache[key]
    try:
        features = build_features(raw, genesis_date=genesis, cfg=run_cfg)
        equity, trades = run_backtest(features, cfg=run_cfg, start_date=start_date)
        bah = buy_and_hold(features, cfg=run_cfg, start_date=start_date)
        bah = bah.reindex(equity.index).ffill()
        bah = bah / bah.iloc[0] * run_cfg.INITIAL_CAPITAL

        n_trades = len(trades)
        win_rate = float((trades["pnl"] > 0).mean()) if n_trades > 0 else float("nan")

        return {
            "buy_z":    run_cfg.BUY_THRESHOLD,
            "sell_z":   run_cfg.SELL_THRESHOLD,
            "z_window": run_cfg.ZSCORE_WINDOW,
            "trend":    run_cfg.USE_TREND_FILTER,
            "volume":   run_cfg.USE_VOLUME_FILTER,
            "macro":    run_cfg.USE_MACRO_FILTER,
            "cagr":     cagr(equity),
            "sharpe":   sharpe_ratio(equity),
            "max_dd":   max_drawdown(equity),
            "calmar":   calmar_ratio(equity),
            "n_trades": n_trades,
            "win_rate": win_rate,
            "bah_cagr": cagr(bah),
        }
    except Exception:
        return None


# ── Pareto front ──────────────────────────────────────────────────────────────

def _is_pareto(results: pd.DataFrame, row_idx: int) -> bool:
    """
    Return True if no other config dominates this one on both
    Sharpe (higher is better) and Max DD (closer to 0 is better).
    """
    row = results.iloc[row_idx]
    for j, other in results.iterrows():
        if j == results.index[row_idx]:
            continue
        if other["sharpe"] >= row["sharpe"] and other["max_dd"] >= row["max_dd"]:
            return True
    return False


def find_pareto_front(results: pd.DataFrame) -> pd.Index:
    """Return index labels of Pareto-optimal rows (max Sharpe, max MaxDD i.e. least negative)."""
    dominated = set()
    idxs = list(results.index)
    for i, a in enumerate(idxs):
        for b in idxs:
            if a == b:
                continue
            # b dominates a if b is at least as good on both metrics and strictly better on one
            if (results.at[b, "sharpe"] >= results.at[a, "sharpe"] and
                    results.at[b, "max_dd"] >= results.at[a, "max_dd"] and
                    (results.at[b, "sharpe"] > results.at[a, "sharpe"] or
                     results.at[b, "max_dd"] > results.at[a, "max_dd"])):
                dominated.add(a)
                break
    return results.index.difference(pd.Index(list(dominated)))


# ── Overfitting flags ─────────────────────────────────────────────────────────

def flag_overfit(results: pd.DataFrame) -> pd.Series:
    """
    Flag configs where n_trades < 5 (too few to be statistically meaningful)
    or where Sharpe is in the top quartile but n_trades < 15 (likely lucky).
    """
    top_sharpe = results["sharpe"].quantile(0.75)
    overfit = (
        (results["n_trades"] < 5) |
        ((results["sharpe"] > top_sharpe) & (results["n_trades"] < 15))
    )
    return overfit


# ── Text report ───────────────────────────────────────────────────────────────

def build_report(
    results: pd.DataFrame,
    pareto_idx: pd.Index,
    overfit: pd.Series,
    symbol: str,
    grid_name: str,
    start_date: str,
    elapsed: float,
    top_n: int,
) -> str:
    lines: list[str] = []
    A = lines.append

    A(f"# {symbol} Parameter Sweep — {date.today()}")
    A("")
    A(f"**Grid:** {grid_name}  ·  **Configs tested:** {len(results)}  "
      f"·  **Period:** {start_date} → today  ·  **Runtime:** {elapsed:.0f}s")
    A("")

    # ── Top N by Sharpe ───────────────────────────────────────────────────────
    A(f"## Top {top_n} Configs by Sharpe Ratio")
    A("")
    A("```")
    A(f"  {'#':<4} {'buyZ':>5} {'sellZ':>6} {'win':>5} {'vol':>5} {'mac':>5} "
      f"{'zWin':>6}  {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} "
      f"{'Trades':>7} {'WinR':>6}  {'⚠':>3}")
    A(f"  {'─'*4} {'─'*5} {'─'*6} {'─'*5} {'─'*5} {'─'*5} "
      f"{'─'*6}  {'─'*7} {'─'*7} {'─'*7} {'─'*7} "
      f"{'─'*7} {'─'*6}  {'─'*3}")

    top = results.nlargest(top_n, "sharpe")
    for rank, (idx, row) in enumerate(top.iterrows(), 1):
        is_p = "★" if idx in pareto_idx else " "
        is_o = "⚠" if overfit.at[idx] else " "
        flag = is_p + is_o
        A(f"  {rank:<4} {row['buy_z']:>5.1f} {row['sell_z']:>6.1f} "
          f"{'Y' if row['trend'] else 'N':>5} {'Y' if row['volume'] else 'N':>5} "
          f"{'Y' if row['macro'] else 'N':>5} {row['z_window']:>6}  "
          f"{row['cagr']*100:>+6.1f}% {row['sharpe']:>7.3f} "
          f"{row['max_dd']*100:>+6.1f}% {row['calmar']:>7.3f} "
          f"{row['n_trades']:>7.0f} {row['win_rate']*100:>5.0f}%  {flag}")
    A("```")
    A("")
    A("★ = Pareto-optimal (best Sharpe for its MaxDD level)")
    A("⚠ = Potential overfit (too few trades for the Sharpe to be meaningful)")
    A("")

    # ── Pareto-optimal configs ────────────────────────────────────────────────
    A("## Pareto-Optimal Configs (Sharpe vs Max Drawdown Frontier)")
    A("")
    A("These configs are not dominated — no other config is both higher Sharpe")
    A("and lower drawdown simultaneously.")
    A("")
    A("```")
    A(f"  {'buyZ':>5} {'sellZ':>6} {'win':>5} {'vol':>5} {'mac':>5} "
      f"{'zWin':>6}  {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} "
      f"{'Trades':>7}")
    A(f"  {'─'*5} {'─'*6} {'─'*5} {'─'*5} {'─'*5} "
      f"{'─'*6}  {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    pareto_rows = results.loc[pareto_idx].sort_values("max_dd", ascending=False)
    for _, row in pareto_rows.iterrows():
        A(f"  {row['buy_z']:>5.1f} {row['sell_z']:>6.1f} "
          f"{'Y' if row['trend'] else 'N':>5} {'Y' if row['volume'] else 'N':>5} "
          f"{'Y' if row['macro'] else 'N':>5} {row['z_window']:>6}  "
          f"{row['cagr']*100:>+6.1f}% {row['sharpe']:>7.3f} "
          f"{row['max_dd']*100:>+6.1f}% {row['calmar']:>7.3f} "
          f"{row['n_trades']:>7.0f}")
    A("```")
    A("")

    # ── Factor analysis ───────────────────────────────────────────────────────
    A("## Factor Analysis — Average Sharpe by Parameter Value")
    A("")
    A("```")
    for col, label in [("buy_z", "Buy Z"), ("sell_z", "Sell Z"),
                        ("z_window", "Z window"), ("trend", "Trend filter"),
                        ("volume", "Volume filter"), ("macro", "Macro filter")]:
        A(f"  {label}:")
        for val, grp in results.groupby(col):
            avg_sh = grp["sharpe"].mean()
            avg_dd = grp["max_dd"].mean() * 100
            bar = "█" * int(max(0, avg_sh * 10))
            A(f"    {str(val):>6}  Sharpe {avg_sh:>5.3f}  MaxDD {avg_dd:>+5.1f}%  {bar}")
        A("")
    A("```")
    A("")

    A("## Default Config Rank")
    A("")
    default_row = results[
        (results["buy_z"]    == cfg.BUY_THRESHOLD) &
        (results["sell_z"]   == cfg.SELL_THRESHOLD) &
        (results["z_window"] == cfg.ZSCORE_WINDOW)  &
        (results["trend"]    == cfg.USE_TREND_FILTER) &
        (results["volume"]   == cfg.USE_VOLUME_FILTER) &
        (results["macro"]    == cfg.USE_MACRO_FILTER)
    ]
    ranked = results["sharpe"].rank(ascending=False)
    if not default_row.empty:
        default_idx = default_row.index[0]
        default_rank = int(ranked.at[default_idx])
        row = default_row.iloc[0]
        A(f"The default config (buy_z={cfg.BUY_THRESHOLD}, sell_z={cfg.SELL_THRESHOLD}, "
          f"window={cfg.ZSCORE_WINDOW}, all filters ON) ranks **#{default_rank}** of {len(results)} "
          f"by Sharpe with Sharpe={row['sharpe']:.3f}, CAGR={row['cagr']*100:+.1f}%, "
          f"MaxDD={row['max_dd']*100:.1f}%.")
    else:
        A("Default config not in sweep grid.")
    A("")

    return "\n".join(lines)


# ── Optional chart ────────────────────────────────────────────────────────────

def plot_frontier(
    results: pd.DataFrame,
    pareto_idx: pd.Index,
    overfit: pd.Series,
    symbol: str,
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping frontier chart")
        return

    BG, FG, GRID = "#0d1117", "#e6edf3", "#21262d"
    GREEN, RED, YELLOW, BLUE, GREY = "#3fb950", "#f85149", "#d29922", "#58a6ff", "#8b949e"

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID,
        "axes.labelcolor": FG, "axes.titlecolor": FG,
        "xtick.color": GREY, "ytick.color": GREY, "text.color": FG,
        "grid.color": GRID, "font.family": "monospace", "font.size": 9,
    })

    fig, ax = plt.subplots(figsize=(12, 7))

    normal = results.loc[~results.index.isin(pareto_idx) & ~overfit]
    pareto = results.loc[pareto_idx]
    over   = results.loc[overfit]

    ax.scatter(normal["max_dd"] * 100, normal["sharpe"],
               color=GREY, alpha=0.4, s=20, label="Configs")
    ax.scatter(over["max_dd"] * 100, over["sharpe"],
               color=YELLOW, alpha=0.5, s=20, marker="x", label="⚠ Overfit risk")
    ax.scatter(pareto["max_dd"] * 100, pareto["sharpe"],
               color=GREEN, s=70, zorder=5, label="★ Pareto-optimal")

    # Connect pareto front
    pf = pareto.sort_values("max_dd")
    ax.plot(pf["max_dd"] * 100, pf["sharpe"],
            color=GREEN, lw=1.0, alpha=0.5, ls="--")

    # Label pareto points
    for _, row in pareto.iterrows():
        label = f"Z={row['buy_z']:.1f}|w={row['z_window']}"
        ax.annotate(label,
                    xy=(row["max_dd"] * 100, row["sharpe"]),
                    xytext=(4, 4), textcoords="offset points",
                    color=GREEN, fontsize=7)

    # Default config
    default = results[
        (results["buy_z"]    == cfg.BUY_THRESHOLD) &
        (results["sell_z"]   == cfg.SELL_THRESHOLD) &
        (results["z_window"] == cfg.ZSCORE_WINDOW)
    ]
    if not default.empty:
        d = default.iloc[0]
        ax.scatter(d["max_dd"] * 100, d["sharpe"],
                   color=BLUE, s=120, marker="D", zorder=6, label="Default config")

    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(f"{symbol}  ·  Sharpe vs Max Drawdown Frontier  ·  {date.today()}", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.invert_xaxis()  # left = less drawdown = better

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frontier_{symbol.lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Frontier chart → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parameter sweep for BTC/ETH swing trading model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC", choices=["BTC", "ETH"])
    p.add_argument("--start",  default=cfg.START_DATE)
    p.add_argument("--quick",  action="store_true",
                   help="Use smaller grid (faster)")
    p.add_argument("--top",    type=int, default=20,
                   help="Show top N configs in the ranked table")
    p.add_argument("--chart",  action="store_true",
                   help="Plot Sharpe vs MaxDD frontier chart")
    p.add_argument("--out",    default=None,
                   help="Path for markdown report")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    symbol    = args.symbol
    yf_symbol = cfg.BTC_SYMBOL if symbol == "BTC" else cfg.ETH_SYMBOL
    genesis   = cfg.BTC_GENESIS if symbol == "BTC" else cfg.ETH_GENESIS
    grid      = QUICK_GRID if args.quick else FULL_GRID
    grid_name = "QUICK" if args.quick else "FULL"

    # Count combos
    combos = list(itertools.product(
        grid["buy_z"], grid["sell_z"], grid["z_window"],
        grid["trend"], grid["volume"], grid["macro"]
    ))
    n_combos = len(combos)

    print(f"\n{'═'*60}")
    print(f"  {symbol} Parameter Sweep  ({grid_name} grid — {n_combos} configs)")
    print(f"{'═'*60}")

    # Fetch data once
    print(f"\n  Fetching {yf_symbol}...", flush=True)
    df = fetch_ohlcv(yf_symbol, start="2014-01-01")
    print(f"  {len(df)} trading days")

    # Pre-build raw feature sets for each unique z_window value
    # (the power-law curve is the same; only Z-score window changes)
    print(f"\n  Pre-computing base features for each Z window...", flush=True)
    features_cache: dict[int, pd.DataFrame] = {}
    for w in sorted(set(grid["z_window"])):
        _dummy_cfg = _make_cfg(1.5, 1.5, w, True, True, True)
        features_cache[w] = df  # pass raw df; build_features handles it per-run

    # Run sweep
    print(f"  Running {n_combos} backtests...\n", flush=True)
    t0 = time.time()
    records = []

    for i, (bz, sz, zw, tr, vo, ma) in enumerate(combos):
        run_cfg = _make_cfg(bz, sz, zw, tr, vo, ma)
        result  = _run_one(features_cache, genesis, run_cfg, args.start)
        if result:
            records.append(result)

        # Progress
        if (i + 1) % max(1, n_combos // 20) == 0 or i == n_combos - 1:
            pct = (i + 1) / n_combos * 100
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_combos - i - 1)
            print(f"  {i+1:>4}/{n_combos}  {pct:>5.1f}%  elapsed {elapsed:.0f}s  "
                  f"eta {eta:.0f}s", end="\r", flush=True)

    elapsed = time.time() - t0
    print(f"\n\n  Completed {len(records)} valid configs in {elapsed:.0f}s")

    if not records:
        print("  No valid results — check data or parameters.")
        return

    results = pd.DataFrame(records)
    results = results.dropna(subset=["sharpe", "max_dd"]).reset_index(drop=True)

    # Pareto front and overfit flags
    pareto_idx = find_pareto_front(results)
    overfit    = flag_overfit(results)

    # Build and print report
    report = build_report(results, pareto_idx, overfit, symbol,
                          grid_name, args.start, elapsed, args.top)
    print("\n" + report)

    # Save report
    out_path = Path(args.out) if args.out else (
        ROOT / "reports" / f"sweep_{symbol}_{date.today()}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\n  Report saved → {out_path}")

    # Optional frontier chart
    if args.chart:
        plot_frontier(results, pareto_idx, overfit, symbol,
                      ROOT / "reports" / "charts")

    # Save raw results as CSV for further analysis
    csv_path = ROOT / "reports" / f"sweep_{symbol}_{date.today()}.csv"
    results.to_csv(csv_path, index=False)
    print(f"  Raw results  → {csv_path}")


if __name__ == "__main__":
    main()
