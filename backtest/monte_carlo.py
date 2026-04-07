"""
backtest/monte_carlo.py — Monte Carlo simulation for strategy robustness.

Two modes
---------
1. Trade resampling
   Shuffle the sequence of closed trades 1,000+ times.
   Each permutation replays as an equity curve → distribution of outcomes.
   Shows whether results are driven by a few lucky trades or are systematic.

2. Parameter sensitivity
   Vary key parameters (fees, slippage, Z thresholds) and record the
   outcome distribution across the grid.

Usage
-----
    from backtest.monte_carlo import run_trade_resample, run_param_sensitivity

    results = run_trade_resample(trades, initial_capital=10_000, n_sims=1000)
    print(results["summary"])

    # Or run from the CLI:
    python3 scripts/run_monte_carlo.py --symbol BTC --sims 2000 --chart
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.metrics import cagr, max_drawdown, sharpe_ratio


# ── Trade resampling ───────────────────────────────────────────────────────────

def run_trade_resample(
    trades: pd.DataFrame,
    initial_capital: float = 10_000.0,
    n_sims: int = 1_000,
    seed: int = 42,
) -> dict:
    """
    Shuffle the trade P&L sequence n_sims times and replay equity curves.

    Each simulation draws trades *with replacement* (bootstrap) from the
    observed trade set, then compounds them sequentially.  This tests
    whether the return distribution is robust or depends on the specific
    ordering of a few large winners.

    Parameters
    ----------
    trades          : trades DataFrame from run_backtest() — must have ``pnl_pct`` column
    initial_capital : starting portfolio value
    n_sims          : number of Monte Carlo paths
    seed            : random seed for reproducibility

    Returns
    -------
    dict with keys:
        sim_results   : DataFrame (n_sims rows × metrics columns)
        summary       : formatted percentile table string
        equity_paths  : ndarray (n_sims × n_trades) of equity curves
        beat_bah_prob : not computed here (needs benchmark); set manually
    """
    if trades.empty or "pnl_pct" not in trades.columns:
        raise ValueError("trades DataFrame must have a 'pnl_pct' column.")

    pnl_pcts = trades["pnl_pct"].dropna().values
    n_trades = len(pnl_pcts)

    if n_trades < 3:
        raise ValueError(f"Need at least 3 trades for Monte Carlo; got {n_trades}.")

    rng = np.random.default_rng(seed)

    # Each row = one sim, each col = one trade's equity multiplier
    draws = rng.choice(pnl_pcts, size=(n_sims, n_trades), replace=True)
    # Equity path: compound (1 + pnl_pct) across trades
    equity_paths = initial_capital * np.cumprod(1.0 + draws, axis=1)

    # Scalar metrics per simulation
    final_equity  = equity_paths[:, -1]
    total_returns = (final_equity / initial_capital) - 1.0

    # Approximate annualised metrics using trade durations
    avg_dur = _avg_duration_days(trades)
    total_days = max(avg_dur * n_trades, 1.0)
    n_years    = total_days / 365.25

    sim_cagr = (final_equity / initial_capital) ** (1.0 / max(n_years, 0.1)) - 1.0

    # Max drawdown per path
    peak    = np.maximum.accumulate(equity_paths, axis=1)
    dd      = (equity_paths - peak) / peak
    sim_mdd = dd.min(axis=1)

    # Win rate per path
    sim_wr  = (draws > 0).mean(axis=1)

    sim_results = pd.DataFrame({
        "final_equity":  final_equity,
        "total_return":  total_returns,
        "cagr_approx":   sim_cagr,
        "max_drawdown":  sim_mdd,
        "win_rate":      sim_wr,
    })

    summary = _percentile_table(sim_results, initial_capital)

    return {
        "sim_results":   sim_results,
        "summary":       summary,
        "equity_paths":  equity_paths,
        "n_trades":      n_trades,
        "n_sims":        n_sims,
        "seed":          seed,
    }


# ── Parameter sensitivity ──────────────────────────────────────────────────────

def run_param_sensitivity(
    features: pd.DataFrame,
    cfg,
    start_date: str | None = None,
    fee_multipliers:      list[float] | None = None,
    slippage_multipliers: list[float] | None = None,
    buy_z_offsets:        list[float] | None = None,
) -> pd.DataFrame:
    """
    Vary key parameters around the base config and record outcome metrics.

    Each combination runs a full backtest.  Returns a DataFrame with one
    row per parameter combination, sorted by Sharpe ratio descending.

    Parameters
    ----------
    features          : feature DataFrame from build_features()
    cfg               : config module (base parameters)
    start_date        : backtest start date
    fee_multipliers   : list of multipliers on cfg.FEE_RATE (default: 0.5,1,2,3)
    slippage_multipliers : list of multipliers on cfg.SLIPPAGE (default: 0.5,1,2,3)
    buy_z_offsets     : list of offsets added to cfg.BUY_THRESHOLD (default: -0.5,0,+0.5)

    Returns
    -------
    DataFrame with columns: fee_mult, slip_mult, buy_z, cagr, sharpe,
                            max_dd, n_trades, label
    """
    import types
    from backtest.engine import run_backtest, buy_and_hold

    fee_multipliers      = fee_multipliers      or [0.5, 1.0, 2.0, 3.0]
    slippage_multipliers = slippage_multipliers or [0.5, 1.0, 2.0, 3.0]
    buy_z_offsets        = buy_z_offsets        or [-0.5, 0.0, 0.5]

    bah = buy_and_hold(features, cfg=cfg, start_date=start_date)
    if not bah.empty:
        bah = bah / bah.iloc[0] * cfg.INITIAL_CAPITAL

    records = []
    for fm in fee_multipliers:
        for sm in slippage_multipliers:
            for bzo in buy_z_offsets:
                # Build a modified config
                override = types.ModuleType("cfg_sensitivity")
                override.__dict__.update({k: v for k, v in vars(cfg).items()
                                          if not k.startswith("__")})
                override.FEE_RATE      = cfg.FEE_RATE * fm
                override.SLIPPAGE      = cfg.SLIPPAGE * sm
                override.BUY_THRESHOLD = cfg.BUY_THRESHOLD + bzo

                try:
                    equity, trades, _open = run_backtest(features, cfg=override,
                                                         start_date=start_date)
                    if equity.empty:
                        continue
                    records.append({
                        "fee_mult":  fm,
                        "slip_mult": sm,
                        "buy_z":     override.BUY_THRESHOLD,
                        "cagr":      cagr(equity),
                        "sharpe":    sharpe_ratio(equity),
                        "max_dd":    max_drawdown(equity),
                        "n_trades":  len(trades),
                        "label":     f"fee×{fm} slip×{sm} buyZ={override.BUY_THRESHOLD:.1f}",
                    })
                except Exception:
                    continue

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("sharpe", ascending=False).reset_index(drop=True)


# ── Chart generation ──────────────────────────────────────────────────────────

def plot_monte_carlo(
    mc_results: dict,
    bah_total_return: float | None = None,
    out_dir: Path | None = None,
    symbol: str = "BTC",
) -> list[Path]:
    """
    Generate three histogram charts and one equity fan chart.

    Charts saved to out_dir (default: reports/charts/):
      mc_returns_<symbol>.png   — distribution of total returns
      mc_drawdown_<symbol>.png  — distribution of max drawdowns
      mc_equity_<symbol>.png    — fan of equity paths (p5/p25/median/p75/p95)

    Returns list of saved Path objects.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  [monte_carlo] matplotlib not available — skipping charts")
        return []

    out_dir = out_dir or (Path(__file__).resolve().parents[1] / "reports" / "charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_results  = mc_results["sim_results"]
    equity_paths = mc_results["equity_paths"]
    n_sims       = mc_results["n_sims"]
    saved: list[Path] = []

    pct_labels = [5, 25, 50, 75, 95]
    colors     = ["#d62728", "#ff7f0e", "#2ca02c", "#ff7f0e", "#d62728"]

    # ── 1. Total return distribution ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    returns_pct = sim_results["total_return"] * 100
    ax.hist(returns_pct, bins=60, color="#4c72b0", edgecolor="white", alpha=0.8)
    for p, c in zip(pct_labels, colors):
        v = np.percentile(returns_pct, p)
        ax.axvline(v, color=c, lw=1.5, ls="--",
                   label=f"p{p}: {v:+.0f}%")
    if bah_total_return is not None:
        ax.axvline(bah_total_return * 100, color="black", lw=2,
                   label=f"B&H: {bah_total_return*100:+.0f}%")
        beat_prob = (sim_results["total_return"] > bah_total_return).mean() * 100
        ax.set_title(f"{symbol} Monte Carlo — Total Return ({n_sims:,} sims)\n"
                     f"P(beat B&H) = {beat_prob:.1f}%", fontsize=12)
    else:
        ax.set_title(f"{symbol} Monte Carlo — Total Return ({n_sims:,} sims)", fontsize=12)
    ax.set_xlabel("Total Return (%)")
    ax.set_ylabel("Simulations")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    p1 = out_dir / f"mc_returns_{symbol.lower()}.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    saved.append(p1)

    # ── 2. Max drawdown distribution ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    mdd_pct = sim_results["max_drawdown"] * 100
    ax.hist(mdd_pct, bins=60, color="#d62728", edgecolor="white", alpha=0.8)
    for p, c in zip(pct_labels, colors):
        v = np.percentile(mdd_pct, p)
        ax.axvline(v, color=c, lw=1.5, ls="--", label=f"p{p}: {v:.0f}%")
    ax.set_title(f"{symbol} Monte Carlo — Max Drawdown ({n_sims:,} sims)", fontsize=12)
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Simulations")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    p2 = out_dir / f"mc_drawdown_{symbol.lower()}.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    saved.append(p2)

    # ── 3. Equity fan chart ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(1, equity_paths.shape[1] + 1)

    # Shaded percentile bands
    p5  = np.percentile(equity_paths, 5,  axis=0)
    p25 = np.percentile(equity_paths, 25, axis=0)
    p50 = np.percentile(equity_paths, 50, axis=0)
    p75 = np.percentile(equity_paths, 75, axis=0)
    p95 = np.percentile(equity_paths, 95, axis=0)

    ax.fill_between(x, p5,  p95, alpha=0.15, color="#4c72b0", label="p5–p95")
    ax.fill_between(x, p25, p75, alpha=0.30, color="#4c72b0", label="p25–p75")
    ax.plot(x, p50, color="#4c72b0", lw=2, label="Median")
    ax.plot(x, p5,  color="#d62728", lw=1, ls="--")
    ax.plot(x, p95, color="#2ca02c", lw=1, ls="--")

    initial = equity_paths[0, 0] / (1 + mc_results["sim_results"]["total_return"].iloc[0] /
                                     equity_paths.shape[1])
    ax.axhline(mc_results["sim_results"]["final_equity"].median(),
               color="grey", lw=0.8, ls=":")

    ax.set_title(f"{symbol} Monte Carlo — Equity Fan ({n_sims:,} sims)", fontsize=12)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=9)
    plt.tight_layout()
    p3 = out_dir / f"mc_equity_{symbol.lower()}.png"
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    saved.append(p3)

    return saved


# ── Internal helpers ───────────────────────────────────────────────────────────

def _avg_duration_days(trades: pd.DataFrame) -> float:
    """Average trade duration in days; fallback to 30 if not computable."""
    if "entry_date" not in trades.columns or "exit_date" not in trades.columns:
        return 30.0
    try:
        durations = (pd.to_datetime(trades["exit_date"])
                     - pd.to_datetime(trades["entry_date"])).dt.days
        return float(durations.mean()) or 30.0
    except Exception:
        return 30.0


def _percentile_table(sim_results: pd.DataFrame, initial_capital: float) -> str:
    """Format a percentile summary table from sim_results DataFrame."""
    pcts   = [5, 10, 25, 50, 75, 90, 95]
    cols   = {
        "total_return":  ("Total Return", True),
        "cagr_approx":   ("CAGR (approx)", True),
        "max_drawdown":  ("Max Drawdown", True),
        "win_rate":      ("Win Rate", True),
        "final_equity":  ("Final Equity", False),
    }
    w_label = 16
    w_col   = 10
    n_cols  = len(pcts)
    sep     = "─" * (w_label + w_col * n_cols + 2)

    header = f"  {'Metric':<{w_label}}" + "".join(f"  {'p'+str(p):>{w_col-2}}" for p in pcts)
    lines  = [sep, header, sep]

    for col_key, (col_label, is_pct) in cols.items():
        if col_key not in sim_results.columns:
            continue
        row_str = f"  {col_label:<{w_label}}"
        for p in pcts:
            v = np.percentile(sim_results[col_key], p)
            if col_key == "final_equity":
                row_str += f"  ${v:>{w_col-3},.0f}"
            elif is_pct:
                row_str += f"  {v*100:>{w_col-2}.1f}%"
            else:
                row_str += f"  {v:>{w_col-2}.2f}"
        lines.append(row_str)

    lines.append(sep)
    return "\n".join(lines)
