"""
backtest/metrics.py — Performance metric calculations.

All functions accept a daily equity curve (pd.Series, dollar values)
or a trades DataFrame returned by the backtest engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Core metrics ───────────────────────────────────────────────────────────────

def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate."""
    rets = daily_returns(equity)
    n_years = len(rets) / 252.0
    if n_years <= 0:
        return float("nan")
    total = float(equity.iloc[-1] / equity.iloc[0])
    return total ** (1.0 / n_years) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def sharpe_ratio(equity: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio (daily risk-free rate assumed 0 by default)."""
    rets = daily_returns(equity)
    excess = rets - risk_free / 252.0
    if excess.std() == 0:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(252))


def sortino_ratio(equity: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    rets = daily_returns(equity)
    excess = rets - risk_free / 252.0
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("nan")
    return float(excess.mean() / downside.std() * np.sqrt(252))


def calmar_ratio(equity: pd.Series) -> float:
    """CAGR / |max drawdown|."""
    mdd = max_drawdown(equity)
    if mdd == 0:
        return float("nan")
    return cagr(equity) / abs(mdd)


def volatility(equity: pd.Series) -> float:
    """Annualised daily volatility."""
    return float(daily_returns(equity).std() * np.sqrt(252))


# ── Trade-level metrics ────────────────────────────────────────────────────────

def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of closed trades that were profitable."""
    if trades.empty or "pnl" not in trades.columns:
        return float("nan")
    closed = trades[trades["pnl"].notna()]
    if closed.empty:
        return float("nan")
    return float((closed["pnl"] > 0).sum() / len(closed))


def profit_factor(trades: pd.DataFrame) -> float:
    """Sum of winning P&L / sum of |losing P&L|."""
    if trades.empty or "pnl" not in trades.columns:
        return float("nan")
    closed = trades[trades["pnl"].notna()]
    gains  = closed.loc[closed["pnl"] > 0, "pnl"].sum()
    losses = closed.loc[closed["pnl"] < 0, "pnl"].sum()
    if losses == 0:
        return float("inf")
    return float(gains / abs(losses))


def avg_trade_duration(trades: pd.DataFrame) -> float:
    """Average holding period in days."""
    if trades.empty or "entry_date" not in trades.columns or "exit_date" not in trades.columns:
        return float("nan")
    closed = trades[trades["exit_date"].notna()].copy()
    closed["duration"] = (pd.to_datetime(closed["exit_date"])
                          - pd.to_datetime(closed["entry_date"])).dt.days
    return float(closed["duration"].mean())


# ── Summary table ──────────────────────────────────────────────────────────────

def summary_table(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    trades: pd.DataFrame,
    label: str = "Strategy",
    bench_label: str = "Buy & Hold",
) -> str:
    """
    Format a side-by-side performance comparison as a text table.

    Parameters
    ----------
    strategy_equity  : daily equity curve for the model
    benchmark_equity : daily equity curve for buy-and-hold
    trades           : trades DataFrame from the backtest engine
    label            : name for the strategy column
    bench_label      : name for the benchmark column
    """
    def _fmt(v: float, pct: bool = True, decimals: int = 1) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  N/A"
        if pct:
            return f"{v*100:+.{decimals}f}%"
        return f"{v:.{decimals}f}"

    rows = [
        ("CAGR",            cagr(strategy_equity),        cagr(benchmark_equity),        True,  1),
        ("Sharpe",          sharpe_ratio(strategy_equity), sharpe_ratio(benchmark_equity), False, 2),
        ("Sortino",         sortino_ratio(strategy_equity),sortino_ratio(benchmark_equity),False, 2),
        ("Calmar",          calmar_ratio(strategy_equity), calmar_ratio(benchmark_equity), False, 2),
        ("Max Drawdown",    max_drawdown(strategy_equity), max_drawdown(benchmark_equity), True,  1),
        ("Ann. Volatility", volatility(strategy_equity),   volatility(benchmark_equity),   True,  1),
        ("Win Rate",        win_rate(trades),              float("nan"),                   True,  1),
        ("Profit Factor",   profit_factor(trades),         float("nan"),                   False, 2),
        ("Avg Duration(d)", avg_trade_duration(trades),    float("nan"),                   False, 1),
        ("Trades",          float(len(trades)),             float("nan"),                   False, 0),
    ]

    w_label = 20
    w_col   = 12
    sep = "─" * (w_label + w_col * 2 + 4)

    lines = [
        sep,
        f"  {'Metric':<{w_label}} {label:>{w_col}} {bench_label:>{w_col}}",
        sep,
    ]
    for name, sv, bv, pct, dec in rows:
        s_str = _fmt(sv, pct=pct, decimals=dec)
        b_str = _fmt(bv, pct=pct, decimals=dec) if not (isinstance(bv, float) and np.isnan(bv)) else "—"
        lines.append(f"  {name:<{w_label}} {s_str:>{w_col}} {b_str:>{w_col}}")
    lines.append(sep)

    start = strategy_equity.index[0].date()
    end   = strategy_equity.index[-1].date()
    final_s = strategy_equity.iloc[-1]
    final_b = benchmark_equity.iloc[-1]
    initial = strategy_equity.iloc[0]
    lines.append(f"  Period: {start} → {end}")
    lines.append(f"  {label}: ${initial:,.0f} → ${final_s:,.0f}"
                 f"  ({(final_s/initial - 1)*100:+.1f}%)")
    lines.append(f"  {bench_label}: ${initial:,.0f} → ${final_b:,.0f}"
                 f"  ({(final_b/initial - 1)*100:+.1f}%)")
    lines.append(sep)

    return "\n".join(lines)
