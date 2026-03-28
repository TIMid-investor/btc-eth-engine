#!/usr/bin/env python3
"""
scripts/run_signals.py — Live signal dashboard for BTC and ETH.

Shows the current Z-score, power-law expected price, deviation from the
growth curve, filter states, and historical performance context.

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
from models.macro_context import fetch_macro, analyze_macro, btc_spy_correlation


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


# ── Historical performance context ────────────────────────────────────────────

def _print_macro_context(macro: dict, btc_corr: float | None) -> None:
    """Display the broad macro environment panel."""
    print(f"")
    print(f"  Macro Environment")
    print(f"  {'─'*56}")

    # SPY
    if macro.get("spy_price") is not None:
        regime = macro["spy_regime"]
        vs200  = macro.get("spy_vs_200d")
        dd     = macro.get("spy_drawdown")
        chg30  = macro.get("spy_30d_chg")
        vs200_str = f"  {vs200*100:+.1f}% vs 200d SMA" if vs200 is not None else ""
        dd_str    = f"  {dd*100:.1f}% from 52w high" if dd is not None else ""
        chg30_str = f"  {chg30*100:+.1f}% (30d)" if chg30 is not None else ""
        print(f"  S&P 500 (SPY):   ${macro['spy_price']:,.0f}  [{regime}]{vs200_str}{dd_str}{chg30_str}")
    else:
        print(f"  S&P 500 (SPY):   unavailable")

    # VIX
    if macro.get("vix_level") is not None:
        print(f"  VIX (fear):      {macro['vix_level']:.1f}  [{macro['vix_label']}]")
    else:
        print(f"  VIX:             unavailable")

    # DXY
    if macro.get("dxy_price") is not None:
        print(f"  Dollar (DXY):    {macro['dxy_price']:.1f}  [{macro['dxy_trend']}]")
    else:
        print(f"  Dollar (DXY):    unavailable")

    # Oil
    if macro.get("oil_level") is not None:
        print(f"  WTI Oil:         ${macro['oil_level']:.0f}  [{macro['oil_label']}]")
    else:
        print(f"  WTI Oil:         unavailable")

    # BTC-SPY correlation
    if btc_corr is not None:
        if btc_corr > 0.65:
            corr_label = "HIGH — macro closely driving crypto"
        elif btc_corr > 0.35:
            corr_label = "MODERATE — partial macro influence"
        else:
            corr_label = "LOW — crypto decoupling from equities"
        print(f"  BTC-SPY corr(60d): {btc_corr:+.2f}  [{corr_label}]")

    # Key signals
    if macro.get("signals"):
        print(f"")
        for sig in macro["signals"][:4]:  # top 4 most relevant
            label = sig[0] if isinstance(sig, tuple) else sig
            print(f"  ⚑  {label}")

    # Overall
    score = macro.get("macro_score", 0)
    label = macro.get("risk_label", "UNKNOWN")
    print(f"")
    print(f"  Macro score: {score:+d}  →  {label}")

    # Contextual note on what this means for crypto
    if score >= 4:
        print(f"  ↳ Risk-off macro is a headwind. Crypto tends to lag equities")
        print(f"    by 30-60 days at turning points. Watch for SPY stabilisation")
        print(f"    before expecting a sustained crypto recovery.")
    elif score >= 2:
        print(f"  ↳ Mixed macro — some headwinds but not a full risk-off regime.")
        print(f"    A Fed pivot signal or SPY recovery would strengthen the crypto bid.")
    else:
        print(f"  ↳ Macro is not a significant headwind at this time.")
    print(f"  {'─'*56}")


def _print_phase_assessment(
    features: pd.DataFrame, symbol: str, cfg, macro_score: int = 0
) -> None:
    """
    Assess whether the current drawdown looks like a dip (temporary) or
    the start of a true extended bear market.

    Signals examined:
      Z velocity    — rate of Z-score change (still falling vs stabilising)
      Z structure   — making new lows vs holding / recovering
      Days in zone  — how long we've been oversold
      Halving cycle — where we are relative to the last BTC halving
      Macro filter  — whether a shock drawdown is active
      Recovery attempts — whether bounces are reclaiming Z=-1.5

    A simple additive score classifies the phase:
      DIP           — short, stabilising, halving-supportive, no new lows
      ACCUMULATION  — extended, uncertain, mixed signals
      BEAR          — accelerating, new lows, structurally deteriorating
    """
    valid = features.dropna(subset=["zscore"])
    if valid.empty:
        return

    z      = valid["zscore"]
    z_now  = float(z.iloc[-1])
    close  = valid["close"]

    # ── Signal 1: Z velocity (5-day rate of change) ───────────────────────────
    z_vel5 = float(z.diff(5).iloc[-1] / 5.0) if len(z) >= 5 else 0.0
    # Positive = Z improving (less oversold). Negative = still falling.

    # ── Signal 2: New Z low in last 30/60 days ────────────────────────────────
    z_30d_min = float(z.iloc[-30:].min()) if len(z) >= 30 else z_now
    z_60d_min = float(z.iloc[-60:].min()) if len(z) >= 60 else z_now
    new_low_30d = z_now <= z_30d_min + 0.05   # within 0.05 of the 30d trough
    new_low_60d = z_now <= z_60d_min + 0.05

    # ── Signal 3: Days in current buy zone ────────────────────────────────────
    thresh = cfg.BUY_THRESHOLD
    in_zone = (z < -thresh).astype(int)
    days_in_zone = 0
    for v in reversed(in_zone.values):
        if v:
            days_in_zone += 1
        else:
            break

    # ── Signal 4: Recovery attempts (bounces above Z=-1.5 then failed) ────────
    # Count how many times Z bounced above -thresh then fell back below in last year
    z_year = z.iloc[-365:] if len(z) >= 365 else z
    crosses_up = ((z_year >= -thresh) & (z_year.shift(1) < -thresh)).sum()
    crosses_dn = ((z_year < -thresh)  & (z_year.shift(1) >= -thresh)).sum()
    failed_recoveries = int(crosses_dn)  # re-entries after bouncing out

    # ── Signal 5: Halving cycle phase ─────────────────────────────────────────
    halving_phase = "UNKNOWN"
    halving_days  = None
    if symbol == "BTC":
        try:
            from models.regime import BTC_HALVINGS
            today_ts = valid.index[-1]
            past_halvings = [h for h in BTC_HALVINGS if h <= today_ts]
            if past_halvings:
                last_halving = max(past_halvings)
                halving_days = (today_ts - last_halving).days
                if halving_days < 365:
                    halving_phase = "POST_EARLY"
                elif halving_days < 730:
                    halving_phase = "POST_LATE"
                else:
                    halving_phase = "LATE_CYCLE"
        except Exception:
            pass

    # ── Signal 6: Macro / volume ──────────────────────────────────────────────
    macro_ok  = bool(valid["macro_ok"].iloc[-1])
    volume_ok = bool(valid["volume_ok"].iloc[-1])
    vol_ratio = float(
        valid["volume"].iloc[-1] / valid["volume"].rolling(30, min_periods=5).mean().iloc[-1]
    ) if "volume" in valid.columns else 1.0

    # ── Scoring ───────────────────────────────────────────────────────────────
    # Positive score = more bearish. Negative = more bullish (dip).
    score = 0

    # Z velocity
    if z_vel5 < -0.02:     score += 2   # accelerating down — bearish
    elif z_vel5 < 0:       score += 1   # still falling, slowly
    elif z_vel5 > 0.02:    score -= 1   # recovering — dip signal

    # New lows
    if new_low_30d:        score += 2
    elif new_low_60d:      score += 1

    # Duration
    if days_in_zone > 200: score += 2
    elif days_in_zone > 90: score += 1
    elif days_in_zone < 30: score -= 1

    # Failed recoveries (structural weakness)
    if failed_recoveries >= 3: score += 2
    elif failed_recoveries >= 1: score += 1

    # Halving cycle (BTC)
    if halving_phase == "POST_EARLY": score -= 2   # historically strong bull
    elif halving_phase == "POST_LATE": score -= 1
    elif halving_phase == "LATE_CYCLE": score += 1

    # Macro
    if not macro_ok:    score += 2
    if not volume_ok:   score += 1
    if vol_ratio < 0.6: score += 1   # drying volume on down move = distribution

    # ── Add macro environment score ───────────────────────────────────────────
    # Macro adds directly to the phase score (risk-off macro extends drawdowns)
    score += macro_score

    # ── Classification ────────────────────────────────────────────────────────
    if score <= 2:
        phase_label = "DIP  — likely temporary, structural supports intact"
        phase_color = "✓"
    elif score <= 6:
        phase_label = "ACCUMULATION  — extended bottom, direction uncertain"
        phase_color = "~"
    else:
        phase_label = "BEAR  — structural deterioration, more downside possible"
        phase_color = "✗"

    # ── Velocity label ────────────────────────────────────────────────────────
    if z_vel5 > 0.02:
        vel_label = f"+{z_vel5:.3f}/day  ↑ recovering"
    elif z_vel5 < -0.02:
        vel_label = f"{z_vel5:.3f}/day  ↓ still falling"
    else:
        vel_label = f"{z_vel5:+.3f}/day  → stabilising"

    print(f"")
    print(f"  Market Phase Assessment")
    print(f"  {'─'*56}")
    print(f"  Z velocity (5d):      {vel_label}")
    if new_low_30d:
        print(f"  New Z low (30d):      YES — at / near trough  ↓")
    elif new_low_60d:
        print(f"  New Z low (60d):      Near recent 60d low  ↓")
    else:
        print(f"  New Z low:            No — holding above recent lows  ↑")
    print(f"  Days in buy zone:     {days_in_zone}  (below Z={-thresh:.1f})")
    print(f"  Failed recoveries:    {failed_recoveries}  (bounced above Z={-thresh:.1f} then fell back)")
    if symbol == "BTC" and halving_days is not None:
        print(f"  Halving cycle:        {halving_phase}  ({halving_days}d since last halving)")
    print(f"  Macro shock filter:   {'CLEAR ✓' if macro_ok else 'TRIGGERED — shock drawdown'}")
    print(f"  Volume vs 30d avg:    {vol_ratio:.2f}×  "
          f"({'thin — watch for capitulation' if vol_ratio < 0.7 else 'normal' if vol_ratio < 1.3 else 'elevated'})")
    print(f"")
    crypto_score = score - macro_score
    print(f"  Phase score: {score:+d}  (crypto signals {crypto_score:+d}  +  macro {macro_score:+d})")
    print(f"  Assessment:  [{phase_color}] {phase_label}")
    print(f"  {'─'*56}")


def _print_dca_ladder(
    features: pd.DataFrame, current_price: float, genesis: str, cfg
) -> None:
    """
    Show a Z-score-based DCA buy ladder.

    Each rung shows what price BTC would need to be at TODAY for Z to reach
    that level.  Allocations are front-loaded toward deeper oversold levels
    where the historical edge is strongest.  Levels above current Z are
    already in the buy zone; levels below are contingency buys if BTC keeps
    falling.
    """
    valid = features.dropna(subset=["zscore", "log_deviation"])
    if valid.empty:
        return
    latest    = valid.iloc[-1]
    z_now     = float(latest["zscore"])
    log_dev   = float(latest["log_deviation"])
    # rolling std from current Z and log_deviation
    r_std     = abs(log_dev / z_now) if abs(z_now) > 0.01 else float(
        valid["log_deviation"].tail(cfg.ZSCORE_WINDOW).std()
    )
    expected  = float(latest["expected_price"])

    # (z_level, % of capital at this rung)
    RUNGS = [
        (-1.5, 10),
        (-2.0, 15),
        (-2.5, 20),
        (-3.0, 25),
        (-3.5, 20),
        (-4.0, 10),
    ]

    # For triggered rungs, find the most recent crossing in the actual Z series
    z_series = features["zscore"].dropna()

    def last_cross_price(z_thresh: float) -> float | None:
        """Price at the most recent downward crossing of z_thresh."""
        crossed = z_series[(z_series < z_thresh) & (z_series.shift(1) >= z_thresh)]
        if crossed.empty:
            return None
        cross_dt = crossed.index[-1]
        return float(features.loc[cross_dt, "close"])

    print(f"")
    print(f"  DCA Buy Ladder  (Z-score staggered entries)")
    print(f"  {'Rung':>5}  {'Z level':>8}  {'Price at Z':>11}  {'Alloc':>6}  {'Cum':>5}  Status")
    print(f"  {'─'*5}  {'─'*8}  {'─'*11}  {'─'*6}  {'─'*5}  {'─'*20}")

    cum           = 0
    filled_alloc  = 0
    filled_wsum   = 0.0

    for rung_i, (z_lvl, pct) in enumerate(RUNGS, 1):
        cum += pct
        # price if Z were exactly at this level right now
        price_at_z = expected * np.exp(z_lvl * r_std)
        triggered  = z_now <= z_lvl

        if triggered:
            cross_p = last_cross_price(z_lvl)
            entry_p = cross_p if cross_p else price_at_z
            filled_alloc += pct
            filled_wsum  += pct * entry_p
            # How far below current price is this rung?
            gap = (price_at_z - current_price) / current_price * 100
            status = f"IN ZONE  (crossed ~${entry_p:,.0f})"
        else:
            gap = (price_at_z - current_price) / current_price * 100
            status = f"if BTC drops {abs(gap):.0f}% → ${price_at_z:,.0f}"

        # Arrow points to the rung closest to current Z
        arrow = " ◄" if abs(z_now - z_lvl) < 0.35 else ""
        print(f"  {rung_i:>5}  Z={z_lvl:>+.1f}   ${price_at_z:>9,.0f}  {pct:>5}%  {cum:>4}%  {status}{arrow}")

    if filled_alloc > 0:
        wavg = filled_wsum / filled_alloc
        gain = (current_price / wavg - 1) * 100
        print(f"")
        print(f"  Triggered rungs: {filled_alloc}% of capital deployed")
        print(f"  Avg entry (triggered): ${wavg:,.0f}  ·  current P&L: {gain:>+.1f}%")
        remaining = 100 - filled_alloc
        if remaining > 0:
            print(f"  Remaining dry powder: {remaining}%  (held for deeper levels)")


def _print_price_targets(features: pd.DataFrame, genesis: str, cfg) -> None:
    """
    Show BTC price that corresponds to each sell Z-score threshold.

    price_target = expected_price(t) × exp(Z_target × rolling_std)

    The expected price rises over time (power law), so the target is shown
    at today, +6 months, and +12 months — letting you see how the sell
    target escalates as the curve keeps climbing.
    """
    from models.power_law import fit_power_law, _days_since

    valid = features.dropna(subset=["zscore", "log_deviation"])
    if valid.empty:
        return

    latest      = valid.iloc[-1]
    z_now       = float(latest["zscore"])
    log_dev_now = float(latest["log_deviation"])
    # Current rolling std: log_dev = Z * std  →  std = log_dev / Z
    if abs(z_now) > 0.01:
        rolling_std = abs(log_dev_now / z_now)
    else:
        # Fallback: compute directly from the trailing window
        rolling_std = float(
            valid["log_deviation"].tail(cfg.ZSCORE_WINDOW).std()
        )

    params = fit_power_law(features["close"], genesis)
    a, b   = params["a"], params["b"]

    genesis_ts = pd.Timestamp(genesis)
    today      = pd.Timestamp.today()

    def expected_at(future_date: pd.Timestamp) -> float:
        t = max(1.0, (future_date - genesis_ts).days)
        return a * t ** b

    def price_at_z(z_target: float, future_date: pd.Timestamp) -> float:
        exp_price = expected_at(future_date)
        return exp_price * np.exp(z_target * rolling_std)

    sell_zs    = [1.0, 1.5, 2.0, 2.5, 3.0]
    horizons   = [("today", today),
                  ("+6 mo", today + pd.DateOffset(months=6)),
                  ("+12 mo", today + pd.DateOffset(months=12)),
                  ("+18 mo", today + pd.DateOffset(months=18))]

    current_price = float(latest["close"])

    print(f"")
    print(f"  Sell Price Targets  (price at each Z threshold, by peak timing)")
    print(f"  {'Sell Z':>7}  {'Today':>12}  {'+6 mo':>12}  {'+12 mo':>12}  {'+18 mo':>12}")
    print(f"  {'─'*7}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
    for sz in sell_zs:
        prices = [price_at_z(sz, dt) for _, dt in horizons]
        gains  = [(p / current_price - 1) * 100 for p in prices]
        row    = f"  Z={sz:>+.1f}  "
        row   += "  ".join(f"${p:>8,.0f} ({g:>+.0f}%)" if g >= 0 else f"${p:>8,.0f} ({g:>.0f}%)"
                            for p, g in zip(prices, gains))
        print(row)
    print(f"  (current price ${current_price:,.0f}  ·  rolling σ = {rolling_std:.3f}  ·  expected = ${expected_at(today):,.0f})")


def _hold_through_cycle_status(
    features: pd.DataFrame, buy_thresh: float, sell_thresh: float
) -> dict:
    """
    Simulate the hold-through-cycle state machine on historical data and
    return where we currently are: IN_POSITION or WAITING_FOR_ENTRY.
    """
    z     = features["zscore"].dropna()
    close = features["close"].reindex(z.index)

    in_position  = False
    entry_date   = None
    entry_z      = None
    last_buy_date = None
    last_buy_z    = None

    for dt, z_val in z.items():
        if np.isnan(z_val):
            continue
        if not in_position:
            if z_val < -buy_thresh:
                in_position  = True
                entry_date   = dt.date()
                entry_z      = z_val
                last_buy_date = dt.date()
                last_buy_z    = z_val
        else:
            if z_val > sell_thresh:
                in_position = False
                entry_date  = None
                entry_z     = None

    status = "IN POSITION — holding through recovery" if in_position else "WAITING — watching for Z < -{:.1f}".format(buy_thresh)
    return {
        "status":         status,
        "entry_date":     entry_date,
        "entry_z":        entry_z,
        "last_buy_date":  last_buy_date,
        "last_buy_z":     last_buy_z,
    }


def _compute_perf_context(features: pd.DataFrame, buy_thresh: float) -> dict:
    """
    Compute historical performance metrics relevant to the current signal.

    Returns a dict with:
      bins          — list of (label, n_trades, win_pct, avg_fwd_ret) per Z bin
      days_in_zone  — consecutive days Z has been < -buy_thresh
      z_percentile  — where current Z sits in the full historical distribution [0-100]
      expected_revert_pct — expected % gain if Z reverts to 0 (deviation collapses)
    """
    z     = features["zscore"].dropna()
    ld    = features["log_deviation"].dropna()
    close = features["close"]

    latest_z  = float(z.iloc[-1])
    latest_ld = float(ld.iloc[-1])

    # Forward return: close N days later vs today (use 90d as representative hold)
    FWD_DAYS = 90
    fwd_ret = close.pct_change(FWD_DAYS).shift(-FWD_DAYS)

    # Z percentile in full history
    z_pct = float((z < latest_z).mean() * 100)

    # Consecutive days Z below threshold
    in_zone = (z < -buy_thresh).astype(int)
    # Walk backwards from the end
    days_in_zone = 0
    for v in reversed(in_zone.values):
        if v:
            days_in_zone += 1
        else:
            break

    # Expected % gain if log-deviation reverts to 0
    expected_revert_pct = float((np.exp(-latest_ld) - 1) * 100)

    # Win-rate by Z-score bin (historical signal days only)
    bins_def = [(-10, -3.0), (-3.0, -2.0), (-2.0, -1.5), (-1.5, -1.0), (-1.0, -0.5)]
    bins_out = []
    for lo, hi in bins_def:
        mask = (z >= lo) & (z < hi)
        sub_fwd = fwd_ret.reindex(z.index)[mask].dropna()
        if len(sub_fwd) < 3:
            continue
        n       = len(sub_fwd)
        win_pct = float((sub_fwd > 0).mean() * 100)
        avg_ret = float(sub_fwd.mean() * 100)
        label   = f"[{lo:+.1f}, {hi:+.1f})"
        bins_out.append((label, n, win_pct, avg_ret))

    return {
        "bins":                 bins_out,
        "days_in_zone":         days_in_zone,
        "z_percentile":         z_pct,
        "expected_revert_pct":  expected_revert_pct,
        "current_z":            latest_z,
        "buy_thresh":           buy_thresh,
    }


def _print_perf_panel(ctx: dict) -> None:
    """Print the historical performance context panel."""
    z      = ctx["current_z"]
    thresh = ctx["buy_thresh"]

    print(f"")
    print(f"  Historical Performance Context  (90-day forward return windows):")
    print(f"  {'Z-score range':<16} {'Signals':>8} {'Win %':>7} {'Avg fwd ret':>12}")
    print(f"  {'─'*16} {'─'*8} {'─'*7} {'─'*12}")
    for label, n, win_pct, avg_ret in ctx["bins"]:
        # Mark the bin that contains the current Z
        marker = " ◄" if _z_in_bin(z, label) else ""
        print(f"  {label:<16} {n:>8} {win_pct:>6.0f}%  {avg_ret:>+10.1f}%{marker}")
    print(f"")
    print(f"  Current Z percentile:   {ctx['z_percentile']:>5.1f}th  "
          f"({'extremely oversold' if ctx['z_percentile'] < 5 else 'deeply oversold' if ctx['z_percentile'] < 15 else 'oversold' if ctx['z_percentile'] < 30 else 'neutral' if ctx['z_percentile'] < 70 else 'overbought'})")
    if ctx["days_in_zone"] > 0:
        print(f"  Days below Z={-thresh:.1f}:       {ctx['days_in_zone']:>5}  days in buy zone")
    print(f"  Expected gain to curve: {ctx['expected_revert_pct']:>+5.1f}%  (if log-deviation reverts to 0)")


def _z_in_bin(z: float, label: str) -> bool:
    """Check if z falls in the bin described by label like '[−2.0, −1.5)'."""
    try:
        parts = label.strip("[]()").split(", ")
        lo, hi = float(parts[0]), float(parts[1])
        return lo <= z < hi
    except Exception:
        return False


# ── Per-symbol dashboard ───────────────────────────────────────────────────────

def print_dashboard(symbol: str, n_rows: int = 1) -> None:
    yf_symbol = cfg.BTC_SYMBOL if symbol == "BTC" else cfg.ETH_SYMBOL
    genesis   = cfg.BTC_GENESIS if symbol == "BTC" else cfg.ETH_GENESIS

    print(f"\n  Fetching {yf_symbol}...", end=" ", flush=True)
    df = fetch_ohlcv(yf_symbol, start="2014-01-01")
    print(f"{len(df)} days", flush=True)

    print(f"  Fetching macro data...", end=" ", flush=True)
    macro_raw  = fetch_macro()
    macro_data = analyze_macro(macro_raw)
    btc_corr   = btc_spy_correlation(df["close"], macro_raw.get("spy"))
    print(f"done", flush=True)

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
    trend_mult  = float(latest.get("trend_mult", 0.5))
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
    trend_dir = "UP" if trend > 0 else "DOWN" if trend < 0 else "FLAT"
    print(f"    Trend (20-wk EMA slope): {trend_dir}  ·  size mult = {trend_mult:.2f}"
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

    # Hold-through-cycle status
    if getattr(cfg, "HOLD_THROUGH_CYCLE", False):
        htc_status = _hold_through_cycle_status(features, cfg.BUY_THRESHOLD, cfg.SELL_THRESHOLD)
        print(f"")
        print(f"  Hold-Through-Cycle mode (buy dip → hold → sell overbought):")
        print(f"    Status:      {htc_status['status']}")
        if htc_status["entry_date"]:
            print(f"    In since:    {htc_status['entry_date']}  (entry Z = {htc_status['entry_z']:+.2f})")
        if htc_status["last_buy_z"]:
            print(f"    Last buy Z:  {htc_status['last_buy_z']:+.2f}  on {htc_status['last_buy_date']}")

    # Macro context
    _print_macro_context(macro_data, btc_corr)

    # Bear phase assessment (macro score feeds in)
    _print_phase_assessment(features, symbol, cfg, macro_data["macro_score"])

    # DCA buy ladder
    _print_dca_ladder(features, price, genesis, cfg)

    # Price targets at each Z level
    _print_price_targets(features, genesis, cfg)

    # Historical performance context
    perf_ctx = _compute_perf_context(features, cfg.BUY_THRESHOLD)
    _print_perf_panel(perf_ctx)

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
