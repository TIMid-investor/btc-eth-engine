#!/usr/bin/env python3
"""
scripts/run_charts.py — Four-panel visual report for BTC or ETH.

Charts produced (saved to reports/charts/):
  1. power_law_<SYMBOL>.png  — log-scale price + fitted curve + ±1σ/2σ bands + trade markers
  2. zscore_<SYMBOL>.png     — Z-score time series with thresholds + entry/exit overlays
  3. equity_<SYMBOL>.png     — strategy vs buy-and-hold (log) + underwater drawdown panel
  4. scatter_<SYMBOL>.png    — entry Z-score vs trade P&L scatter

Usage:
  python3 scripts/run_charts.py                  # both assets
  python3 scripts/run_charts.py --symbol BTC
  python3 scripts/run_charts.py --symbol ETH --start 2019-01-01
  python3 scripts/run_charts.py --symbol BTC --buy-z 2.0 --no-trend
"""

from __future__ import annotations

import argparse
import sys
import types
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import config as cfg
from data.fetcher import fetch_ohlcv
from backtest.engine import build_features, run_backtest, buy_and_hold
from models.power_law import fit_power_law, _days_since


# ── Style ──────────────────────────────────────────────────────────────────────

BG       = "#0d1117"
FG       = "#e6edf3"
GRID     = "#21262d"
GREEN    = "#3fb950"
RED      = "#f85149"
YELLOW   = "#d29922"
BLUE     = "#58a6ff"
PURPLE   = "#bc8cff"
ORANGE   = "#ffa657"
GREY     = "#8b949e"

def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   FG,
        "axes.titlecolor":   FG,
        "xtick.color":       GREY,
        "ytick.color":       GREY,
        "text.color":        FG,
        "grid.color":        GRID,
        "grid.linewidth":    0.6,
        "legend.facecolor":  "#161b22",
        "legend.edgecolor":  GRID,
        "legend.labelcolor": FG,
        "font.family":       "monospace",
        "font.size":         9,
    })

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Chart 1: Power-law curve ───────────────────────────────────────────────────

def chart_power_law(
    features: pd.DataFrame,
    trades: pd.DataFrame,
    symbol: str,
    run_cfg,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    prices   = features["close"]
    expected = features["expected_price"]
    log_dev  = features["log_deviation"]

    # Rolling σ for bands
    rolling_std = log_dev.rolling(window=run_cfg.ZSCORE_WINDOW,
                                  min_periods=run_cfg.ZSCORE_MIN_PERIODS).std()
    upper2 = expected * np.exp(2 * rolling_std)
    upper1 = expected * np.exp(1 * rolling_std)
    lower1 = expected * np.exp(-1 * rolling_std)
    lower2 = expected * np.exp(-2 * rolling_std)

    idx = features.index

    # Bands
    ax.fill_between(idx, lower2, lower1, alpha=0.12, color=GREEN,  label="−1σ / −2σ band")
    ax.fill_between(idx, lower1, upper1, alpha=0.10, color=BLUE,   label="±1σ band")
    ax.fill_between(idx, upper1, upper2, alpha=0.12, color=RED,    label="+1σ / +2σ band")

    # Curve and price
    ax.plot(idx, expected, color=BLUE,   lw=1.5,  alpha=0.8, label="Power-law curve")
    ax.plot(idx, prices,   color=ORANGE, lw=1.0,  alpha=0.9, label=f"{symbol} price")

    # Trade markers
    if not trades.empty:
        for _, t in trades.iterrows():
            ed = pd.Timestamp(t["entry_date"])
            xd = pd.Timestamp(t["exit_date"]) if pd.notna(t.get("exit_date")) else None
            ep = t.get("entry_price", np.nan)
            xp = t.get("exit_price",  np.nan)
            if ed in features.index and not np.isnan(ep):
                ax.scatter(ed, ep, marker="^", color=GREEN, s=60, zorder=5, alpha=0.85)
            if xd is not None and xd in features.index and not np.isnan(xp):
                color = GREEN if t.get("pnl", 0) >= 0 else RED
                ax.scatter(xd, xp, marker="v", color=color, s=60, zorder=5, alpha=0.85)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v:,.0f}" if v >= 1 else f"${v:.2f}"
    ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, which="both", alpha=0.4)
    ax.set_title(f"{symbol}  ·  Power-Law Growth Curve  ·  {date.today()}", fontsize=12, pad=12)
    ax.set_ylabel("Price (log scale, USD)")

    # Curve equation annotation
    if run_cfg.CURVE_MODEL == "power_law":
        genesis = cfg.BTC_GENESIS if symbol == "BTC" else cfg.ETH_GENESIS
        params  = fit_power_law(features["close"], genesis)
        ann = (f"price = {params['a']:.2e} × days^{params['b']:.3f}   "
               f"R²={params['r_squared']:.3f}")
        ax.annotate(ann, xy=(0.02, 0.04), xycoords="axes fraction",
                    color=GREY, fontsize=8)

    # Legend: add entry/exit markers manually
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=GREEN,  markersize=8, label="Entry (long)"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=GREEN,  markersize=8, label="Exit (profit)"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=RED,    markersize=8, label="Exit (loss)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir / f"power_law_{symbol.lower()}.png")


# ── Chart 2: Z-score panel ────────────────────────────────────────────────────

def chart_zscore(
    features: pd.DataFrame,
    trades: pd.DataFrame,
    symbol: str,
    run_cfg,
    out_dir: Path,
) -> None:
    fig, (ax_price, ax_z) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.06}
    )

    idx    = features.index
    prices = features["close"]
    z      = features["zscore"]

    # Top: price (log)
    ax_price.plot(idx, prices, color=ORANGE, lw=1.0)
    ax_price.set_yscale("log")
    ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v:,.0f}"
    ))
    ax_price.grid(True, alpha=0.3)
    ax_price.set_ylabel("Price (log)")
    ax_price.set_title(f"{symbol}  ·  Z-Score Signal Dashboard  ·  {date.today()}", fontsize=12, pad=10)

    # Bottom: Z-score
    # Colour-coded background zones
    ax_z.axhspan( run_cfg.BUY_THRESHOLD,  6.0,  alpha=0.08, color=RED,   zorder=0)
    ax_z.axhspan(-6.0, -run_cfg.BUY_THRESHOLD,  alpha=0.08, color=GREEN, zorder=0)
    ax_z.axhspan(-run_cfg.EXIT_THRESHOLD, run_cfg.EXIT_THRESHOLD, alpha=0.06, color=GREY, zorder=0)

    # Threshold lines
    for level, color, label in [
        ( run_cfg.SELL_THRESHOLD, RED,   f"Sell threshold +{run_cfg.SELL_THRESHOLD}"),
        ( 3.0,                    RED,   "+3σ"),
        (-run_cfg.BUY_THRESHOLD,  GREEN, f"Buy threshold −{run_cfg.BUY_THRESHOLD}"),
        (-3.0,                    GREEN, "−3σ"),
        ( 0.0,                    GREY,  ""),
    ]:
        ax_z.axhline(level, color=color, lw=0.8, ls="--", alpha=0.7,
                     label=label if label else None)

    # Z-score line coloured by zone
    z_vals  = z.values
    z_green = np.where(z_vals < -run_cfg.BUY_THRESHOLD, z_vals, np.nan)
    z_red   = np.where(z_vals >  run_cfg.SELL_THRESHOLD, z_vals, np.nan)
    z_grey  = np.where((z_vals >= -run_cfg.BUY_THRESHOLD) & (z_vals <= run_cfg.SELL_THRESHOLD), z_vals, np.nan)

    ax_z.plot(idx, z_grey,  color=GREY,  lw=0.9, alpha=0.8)
    ax_z.plot(idx, z_green, color=GREEN, lw=1.2, alpha=0.9)
    ax_z.plot(idx, z_red,   color=RED,   lw=1.2, alpha=0.9)

    # Trade markers on Z panel
    if not trades.empty:
        for _, t in trades.iterrows():
            ed = pd.Timestamp(t["entry_date"])
            ez = t.get("entry_z", np.nan)
            xd = pd.Timestamp(t["exit_date"]) if pd.notna(t.get("exit_date")) else None
            xz = t.get("exit_z", np.nan)
            if ed in features.index and not np.isnan(ez):
                ax_z.scatter(ed, ez, marker="^", color=GREEN, s=50, zorder=5)
                ax_price.axvline(ed, color=GREEN, lw=0.4, alpha=0.4)
            if xd is not None and not np.isnan(xz):
                c = GREEN if t.get("pnl", 0) >= 0 else RED
                ax_z.scatter(xd, xz, marker="v", color=c, s=50, zorder=5)
                ax_price.axvline(xd, color=c, lw=0.4, alpha=0.3)

    ax_z.set_ylim(-6, 6)
    ax_z.set_ylabel("Z-score")
    ax_z.grid(True, alpha=0.3)
    ax_z.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_z.xaxis.set_major_locator(mdates.YearLocator())
    ax_z.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir / f"zscore_{symbol.lower()}.png")


# ── Chart 3: Equity curve + drawdown ─────────────────────────────────────────

def chart_equity(
    equity: pd.Series,
    bah: pd.Series,
    trades: pd.DataFrame,
    symbol: str,
    run_cfg,
    out_dir: Path,
) -> None:
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
    )

    # Equity curves
    ax_eq.plot(equity.index, equity.values, color=BLUE,   lw=1.5, label=f"{symbol} Strategy")
    ax_eq.plot(bah.index,    bah.values,    color=ORANGE, lw=1.0, alpha=0.7, label="Buy & Hold")
    ax_eq.set_yscale("log")
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax_eq.grid(True, alpha=0.3)
    ax_eq.legend(loc="upper left", fontsize=9)
    ax_eq.set_ylabel("Portfolio value (log, USD)")
    ax_eq.set_title(f"{symbol}  ·  Strategy vs Buy-and-Hold  ·  {date.today()}", fontsize=12, pad=10)

    # Summary annotations
    from backtest.metrics import cagr, sharpe_ratio, max_drawdown
    strat_cagr = cagr(equity)
    bah_cagr   = cagr(bah)
    strat_mdd  = max_drawdown(equity)
    bah_mdd    = max_drawdown(bah)
    strat_sh   = sharpe_ratio(equity)
    ann = (f"Strategy: CAGR {strat_cagr*100:+.1f}%  Sharpe {strat_sh:.2f}  MaxDD {strat_mdd*100:.1f}%\n"
           f"B&H:      CAGR {bah_cagr*100:+.1f}%                  MaxDD {bah_mdd*100:.1f}%")
    ax_eq.annotate(ann, xy=(0.02, 0.04), xycoords="axes fraction",
                   color=GREY, fontsize=8, family="monospace")

    # Trade markers on equity
    if not trades.empty:
        for _, t in trades.iterrows():
            ed = pd.Timestamp(t["entry_date"])
            if ed in equity.index:
                ax_eq.axvline(ed, color=GREEN, lw=0.4, alpha=0.35)

    # Drawdown panel
    def _dd_series(s: pd.Series) -> pd.Series:
        roll_max = s.cummax()
        return (s - roll_max) / roll_max * 100

    strat_dd = _dd_series(equity)
    bah_dd   = _dd_series(bah)

    ax_dd.fill_between(strat_dd.index, strat_dd.values, 0, alpha=0.5, color=BLUE,   label="Strategy DD")
    ax_dd.fill_between(bah_dd.index,   bah_dd.values,   0, alpha=0.3, color=ORANGE, label="B&H DD")
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_dd.grid(True, alpha=0.3)
    ax_dd.legend(loc="lower right", fontsize=8)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())

    fig.tight_layout()
    _save(fig, out_dir / f"equity_{symbol.lower()}.png")


# ── Chart 4: Trade scatter ────────────────────────────────────────────────────

def chart_scatter(
    trades: pd.DataFrame,
    symbol: str,
    out_dir: Path,
) -> None:
    if trades.empty or "entry_z" not in trades.columns:
        print(f"  [scatter] No trades to plot for {symbol}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    pnl_pct = trades["pnl_pct"].fillna(0) * 100
    ez      = trades["entry_z"].fillna(0)
    colors  = [GREEN if p >= 0 else RED for p in pnl_pct]
    sizes   = np.abs(pnl_pct).clip(1, 50) * 8

    sc = ax.scatter(ez, pnl_pct, c=colors, s=sizes, alpha=0.75, edgecolors=GRID, lw=0.5)

    # Zero lines
    ax.axhline(0, color=GREY, lw=0.8, ls="--", alpha=0.6)
    ax.axvline(-cfg.BUY_THRESHOLD,  color=GREEN, lw=0.8, ls="--", alpha=0.6,
               label=f"Buy threshold −{cfg.BUY_THRESHOLD}")
    ax.axvline(-cfg.SELL_THRESHOLD, color=RED,   lw=0.8, ls="--", alpha=0.6,
               label=f"Sell threshold +{cfg.SELL_THRESHOLD}")

    # Trend line
    valid = trades[trades["entry_z"].notna() & trades["pnl_pct"].notna()]
    if len(valid) >= 3:
        coeffs = np.polyfit(valid["entry_z"], valid["pnl_pct"] * 100, 1)
        x_line = np.linspace(ez.min(), ez.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), color=YELLOW, lw=1.2,
                ls="--", alpha=0.6, label="Linear trend")

    ax.set_xlabel("Entry Z-score")
    ax.set_ylabel("Trade P&L (%)")
    ax.set_title(f"{symbol}  ·  Entry Z-Score vs Trade P&L  ·  {date.today()}", fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)

    # Win/loss summary
    wins   = (pnl_pct >= 0).sum()
    losses = (pnl_pct < 0).sum()
    ax.annotate(f"Wins: {wins}  Losses: {losses}  Win rate: {wins/(wins+losses)*100:.0f}%",
                xy=(0.02, 0.96), xycoords="axes fraction", color=GREY, fontsize=8)

    handles = [
        Patch(facecolor=GREEN, label="Winning trade"),
        Patch(facecolor=RED,   label="Losing trade"),
    ] + ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir / f"scatter_{symbol.lower()}.png")


# ── CLI + orchestration ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate 4 visual charts for BTC or ETH",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol",   default=None, choices=["BTC", "ETH"],
                   help="Asset to chart (default: both)")
    p.add_argument("--start",    default=cfg.START_DATE)
    p.add_argument("--end",      default=str(date.today()))
    p.add_argument("--buy-z",    type=float, default=cfg.BUY_THRESHOLD)
    p.add_argument("--sell-z",   type=float, default=cfg.SELL_THRESHOLD)
    p.add_argument("--no-trend", action="store_true")
    p.add_argument("--no-volume",action="store_true")
    p.add_argument("--no-macro", action="store_true")
    p.add_argument("--curve",    default=cfg.CURVE_MODEL, choices=["power_law","log_ema"])
    return p.parse_args()


def _override_cfg(args: argparse.Namespace) -> types.ModuleType:
    import types as _t
    oc = _t.ModuleType("cfg_override")
    oc.__dict__.update({k: v for k, v in vars(cfg).items() if not k.startswith("__")})
    oc.BUY_THRESHOLD     = args.buy_z
    oc.SELL_THRESHOLD    = args.sell_z
    oc.USE_TREND_FILTER  = not args.no_trend
    oc.USE_VOLUME_FILTER = not args.no_volume
    oc.USE_MACRO_FILTER  = not args.no_macro
    oc.CURVE_MODEL       = args.curve
    return oc


def run_symbol(symbol: str, args: argparse.Namespace, run_cfg) -> None:
    yf_symbol = cfg.BTC_SYMBOL if symbol == "BTC" else cfg.ETH_SYMBOL
    genesis   = cfg.BTC_GENESIS if symbol == "BTC" else cfg.ETH_GENESIS
    out_dir   = ROOT / "reports" / "charts"

    print(f"\n  [{symbol}] Fetching data...", flush=True)
    df = fetch_ohlcv(yf_symbol, start="2014-01-01", end=args.end)

    print(f"  [{symbol}] Building features...", flush=True)
    features = build_features(df, genesis_date=genesis, cfg=run_cfg)

    print(f"  [{symbol}] Running backtest...", flush=True)
    equity, trades = run_backtest(features, cfg=run_cfg, start_date=args.start)
    bah = buy_and_hold(features, cfg=run_cfg, start_date=args.start)
    bah = bah.reindex(equity.index).ffill()
    bah = bah / bah.iloc[0] * run_cfg.INITIAL_CAPITAL

    _style()
    print(f"  [{symbol}] Chart 1/4  power-law curve...")
    chart_power_law(features, trades, symbol, run_cfg, out_dir)
    print(f"  [{symbol}] Chart 2/4  Z-score panel...")
    chart_zscore(features, trades, symbol, run_cfg, out_dir)
    print(f"  [{symbol}] Chart 3/4  equity curve...")
    chart_equity(equity, bah, trades, symbol, run_cfg, out_dir)
    print(f"  [{symbol}] Chart 4/4  trade scatter...")
    chart_scatter(trades, symbol, out_dir)


def main() -> None:
    args    = parse_args()
    run_cfg = _override_cfg(args)
    symbols = [args.symbol] if args.symbol else ["BTC", "ETH"]
    print(f"\n  Crypto Charts — {date.today()}")
    for sym in symbols:
        run_symbol(sym, args, run_cfg)
    print(f"\n  Done. Charts saved to {ROOT / 'reports' / 'charts'}/")


if __name__ == "__main__":
    main()
