#!/usr/bin/env python3
"""
scripts/run_forecast.py — Probabilistic price forecast with historical analogs.

Models the Z-score as an Ornstein-Uhlenbeck (mean-reverting) process, fits
the reversion speed and daily volatility to historical data, then simulates
1 000 forward paths.  Converts Z paths back to price paths using the
power-law expected price curve projected forward.

Also overlays the 3–4 most relevant historical periods that started at a
similar Z-score entry point, showing how price actually evolved from those
analogous bottoms.

Chart saved to: reports/charts/forecast_{SYMBOL}_{DATE}.png

Usage:
  python3 scripts/run_forecast.py
  python3 scripts/run_forecast.py --symbol ETH
  python3 scripts/run_forecast.py --symbol BTC --horizon 24 --paths 2000
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config as cfg
from data.fetcher import fetch_ohlcv
from backtest.engine import build_features
from models.power_law import fit_power_law, _days_since


# ── Style (matches run_charts.py) ─────────────────────────────────────────────

BG     = "#0d1117"
FG     = "#e6edf3"
GRID   = "#21262d"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
BLUE   = "#58a6ff"
PURPLE = "#bc8cff"
ORANGE = "#ffa657"
GREY   = "#8b949e"

ANALOG_COLORS = ["#58a6ff", "#3fb950", "#ffa657", "#bc8cff", "#f85149"]


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor":   BG,
        "axes.edgecolor":   GRID, "axes.labelcolor": FG,
        "axes.titlecolor":  FG, "xtick.color":      GREY,
        "ytick.color":      GREY, "text.color":      FG,
        "grid.color":       GRID, "grid.linewidth":  0.6,
        "legend.facecolor": "#161b22", "legend.edgecolor": GRID,
        "legend.labelcolor": FG, "font.family":     "monospace",
        "font.size": 9,
    })


# ── OU process ────────────────────────────────────────────────────────────────

def fit_ou(z: pd.Series) -> dict:
    """
    Fit a discrete-time OU process to the Z-score history.

    Z_{t+1} = phi * Z_t + sigma * eps,  eps ~ N(0,1)

    Uses AR(1) OLS.  The mean-reversion speed theta = 1 - phi (daily).
    Half-life in days = ln(2) / theta.
    """
    z = z.dropna()
    # Remove any residual infinities that would break lstsq
    z = z[np.isfinite(z)]

    dz    = z.diff().dropna()
    z_lag = z.shift(1).reindex(dz.index)

    # Drop any NaN/inf pairs
    mask  = np.isfinite(dz.values) & np.isfinite(z_lag.values)
    dz_v  = dz.values[mask]
    zl_v  = z_lag.values[mask]

    X    = np.column_stack([np.ones(len(zl_v)), zl_v])
    coef = np.linalg.lstsq(X, dz_v, rcond=None)[0]

    theta     = float(max(-coef[1], 1e-4))
    phi       = 1.0 - theta
    pred      = X @ coef
    residuals = dz_v - pred
    sigma     = float(np.std(residuals[np.isfinite(residuals)]))
    half_life = np.log(2) / theta

    return dict(theta=theta, phi=phi, sigma=sigma, half_life=half_life)


def simulate_z(z0: float, ou: dict, n_days: int, n_paths: int = 1000,
               seed: int = 42) -> np.ndarray:
    """
    Simulate n_paths forward Z-score paths starting at z0.
    Returns array (n_paths, n_days+1) — index 0 is today.
    """
    rng   = np.random.default_rng(seed)
    paths = np.zeros((n_paths, n_days + 1))
    paths[:, 0] = z0
    noise = rng.normal(0, ou["sigma"], (n_paths, n_days))
    phi   = ou["phi"]
    for t in range(n_days):
        paths[:, t + 1] = phi * paths[:, t] + noise[:, t]
    return paths


def z_to_price(z_paths: np.ndarray, future_dates: pd.DatetimeIndex,
               a: float, b: float, genesis: str, log_std: float) -> np.ndarray:
    """
    Convert Z-score path array (n_paths, n_days+1) to price paths.

    price = a * t^b * exp(Z * log_std)

    log_std is the current rolling std of log-deviations — held constant
    for the forecast period (conservative / stable assumption).
    """
    origin = pd.Timestamp(genesis)
    t      = np.maximum(1.0, (future_dates - origin).days.values.astype(float))
    exp_px = a * t ** b                                   # shape (n_days+1,)
    return exp_px[np.newaxis, :] * np.exp(z_paths * log_std)


# ── Historical analogues ──────────────────────────────────────────────────────

def find_analogs(features: pd.DataFrame,
                 z_lo: float = -3.5, z_hi: float = -1.8,
                 fwd_days: int = 548,
                 n: int = 4,
                 min_year: int = 2018) -> list[dict]:
    """
    Find the N most recent historical bear-market periods where Z entered
    the [z_lo, z_hi] window and return the subsequent price path, normalised
    so that the entry price = today's current price.

    Each analog dict contains:
      entry_date  : when Z first crossed below z_hi from above
      entry_price : actual price at that date
      norm_factor : today_price / entry_price  (for normalising to today)
      prices      : pd.Series of actual prices for the next fwd_days
      z_path      : pd.Series of Z-scores for the same window
    """
    z     = features["zscore"].dropna()
    close = features["close"]
    today_price = float(close.iloc[-1])

    # Find each downward crossing of z_hi
    in_zone     = (z <= z_hi).astype(int)
    crossings   = in_zone[(in_zone == 1) & (in_zone.shift(1) == 0)]

    # Keep only entries deep enough (Z reaches z_lo at some point after)
    analogs = []
    for cross_dt in crossings.index:
        idx_start = features.index.get_loc(cross_dt)
        idx_end   = min(idx_start + fwd_days, len(features) - 1)
        window_z  = z.iloc[idx_start:idx_end + 1]

        if window_z.min() > z_lo:       # never got very deep — skip
            continue
        if idx_end - idx_start < 90:    # too short a window to be useful
            continue
        # Don't include periods that are "right now" (within 30 days of today)
        if (features.index[-1] - cross_dt).days < 30:
            continue
        # Exclude very early market history (incomparable market structure)
        if cross_dt.year < min_year:
            continue

        ep    = float(close.loc[cross_dt])
        nf    = today_price / ep
        pslice = close.iloc[idx_start:idx_end + 1]
        zslice = z.reindex(pslice.index)

        analogs.append(dict(
            entry_date  = cross_dt,
            entry_z     = float(z.loc[cross_dt]),
            entry_price = ep,
            norm_factor = nf,
            prices      = pslice,
            z_path      = zslice,
            days        = idx_end - idx_start,
        ))

    # Return the N most recent
    return analogs[-n:]


# ── Probability table ─────────────────────────────────────────────────────────

def prob_table(price_paths: np.ndarray, future_dates: pd.DatetimeIndex,
               current_price: float, horizons_mo: list[int]) -> list[dict]:
    """
    For each horizon, compute:
      median price, mean price, various percentiles,
      P(price > 1.5x, 2x, 3x), P(price < 0.7x).
    """
    today = future_dates[0]
    rows  = []
    for mo in horizons_mo:
        target_dt = today + pd.DateOffset(months=mo)
        # Find nearest index
        idx = int(np.argmin(np.abs((future_dates - target_dt).days)))
        col = price_paths[:, idx]
        rows.append(dict(
            months        = mo,
            date          = future_dates[idx].date(),
            p5            = float(np.percentile(col, 5)),
            p25           = float(np.percentile(col, 25)),
            median        = float(np.percentile(col, 50)),
            p75           = float(np.percentile(col, 75)),
            p95           = float(np.percentile(col, 95)),
            pct_above_15x = float((col > current_price * 1.5).mean() * 100),
            pct_above_2x  = float((col > current_price * 2.0).mean() * 100),
            pct_above_3x  = float((col > current_price * 3.0).mean() * 100),
            pct_below_07x = float((col < current_price * 0.70).mean() * 100),
        ))
    return rows


# ── Chart ─────────────────────────────────────────────────────────────────────

def make_chart(
    features:      pd.DataFrame,
    z_paths:       np.ndarray,
    price_paths:   np.ndarray,
    future_dates:  pd.DatetimeIndex,
    analogs:       list[dict],
    params:        dict,
    genesis:       str,
    current_price: float,
    current_z:     float,
    log_std:       float,
    ou_params:     dict,
    prob_rows:     list[dict],
    symbol:        str,
    out_path:      Path,
) -> None:
    _style()

    fig = plt.figure(figsize=(15, 11))
    # Layout: price panel (top), z-score panel (middle), prob table text (bottom)
    gs  = fig.add_gridspec(3, 1, height_ratios=[5, 2, 1.6], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")

    today     = future_dates[0]
    hist_start = today - pd.DateOffset(years=3)
    hist      = features[features.index >= hist_start]

    # ── Power-law expected price (history + forward) ──────────────────────────
    origin = pd.Timestamp(genesis)
    a, b   = params["a"], params["b"]

    all_dates = hist.index.append(future_dates[1:])
    t_all     = np.maximum(1.0, (all_dates - origin).days.values.astype(float))
    curve_all = a * t_all ** b

    ax1.semilogy(all_dates, curve_all, color=YELLOW, lw=1.2, ls="--",
                 alpha=0.7, label="Power-law expected", zorder=2)

    # ±1σ / ±2σ bands on the forward portion only
    fwd_t     = np.maximum(1.0, (future_dates - origin).days.values.astype(float))
    curve_fwd = a * fwd_t ** b
    for n_sig, alpha, color in [(1, 0.12, GREEN), (2, 0.07, YELLOW)]:
        upper = curve_fwd * np.exp( n_sig * log_std)
        lower = curve_fwd * np.exp(-n_sig * log_std)
        ax1.fill_between(future_dates, lower, upper,
                         color=color, alpha=alpha, zorder=1)

    # ── Monte Carlo price fan ─────────────────────────────────────────────────
    pcts = np.percentile(price_paths, [5, 25, 50, 75, 95], axis=0)
    ax1.fill_between(future_dates, pcts[0], pcts[4],
                     color=BLUE, alpha=0.08, zorder=3, label="5–95th pct (path uncertainty)")
    ax1.fill_between(future_dates, pcts[1], pcts[3],
                     color=BLUE, alpha=0.16, zorder=4, label="25–75th pct")
    ax1.semilogy(future_dates, pcts[2], color=BLUE, lw=1.5, ls="--",
                 alpha=0.9, zorder=5, label="Median path")

    # ── Parameter uncertainty band ────────────────────────────────────────────
    # The 95% CI on the OU mean-reversion speed θ from a single ~3000-day
    # history is approximately ±50%.  Simulate the SLOW reversion scenario
    # (θ × 0.5, i.e. half-life doubles) and the FAST scenario (θ × 1.5).
    # The outer envelope of these median paths shows how much the fan chart
    # would shift under plausible parameter uncertainty — a band the inner
    # fan chart does not capture.
    n_days_fwd = len(future_dates) - 1
    n_paths_pu = max(200, len(z_paths) // 5)   # smaller set — just need medians
    for theta_mult, label_str, seed_offset in [
        (0.5, "Param uncertainty (slow reversion, θ×0.5)", 99),
        (1.5, "Param uncertainty (fast reversion, θ×1.5)", 199),
    ]:
        ou_pu    = dict(ou_params)
        ou_pu["theta"]     = ou_params["theta"] * theta_mult
        ou_pu["phi"]       = 1.0 - ou_pu["theta"]
        ou_pu["half_life"] = np.log(2) / max(ou_pu["theta"], 1e-6)
        try:
            z_pu     = simulate_z(current_z, ou_pu, n_days_fwd,
                                  n_paths=n_paths_pu, seed=42 + seed_offset)
            px_pu    = z_to_price(z_pu, future_dates, a, b, genesis, log_std)
            med_pu   = np.median(px_pu, axis=0)
            ax1.semilogy(future_dates, med_pu, color=GREY, lw=0.9, ls=":",
                         alpha=0.6, zorder=3, label=label_str)
        except Exception:
            pass
    # Add a visual hint to the legend explaining the distinction
    ax1.plot([], [], color=GREY, lw=0.9, ls=":",
             label="(dotted = ±50% θ parameter uncertainty)")

    # ── Historical price ──────────────────────────────────────────────────────
    ax1.semilogy(hist.index, hist["close"], color=FG, lw=1.4,
                 alpha=0.95, zorder=6, label=f"{symbol} price")

    # ── Historical analog paths ───────────────────────────────────────────────
    for i, analog in enumerate(analogs):
        color  = ANALOG_COLORS[i % len(ANALOG_COLORS)]
        scaled = analog["prices"] * analog["norm_factor"]
        label  = f"Analog {analog['entry_date'].strftime('%b %Y')} (×{analog['norm_factor']:.1f})"
        # Shift analog dates so entry aligns with today
        offset = today - analog["entry_date"]
        shifted_idx = analog["prices"].index + offset
        ax1.semilogy(shifted_idx, scaled.values, color=color, lw=1.1,
                     alpha=0.55, ls="-", zorder=5, label=label)

    # ── Today marker ─────────────────────────────────────────────────────────
    ax1.axvline(today, color=GREY, lw=0.8, ls=":", alpha=0.7, zorder=7)
    ax1.scatter([today], [current_price], color=RED, s=60, zorder=10,
                label=f"Now: ${current_price:,.0f}")

    # Clamp y-axis: MC 3rd–97th pct keeps analogs from blowing the scale
    p03 = float(np.percentile(price_paths[:, 1:], 3))
    p97 = float(np.percentile(price_paths[:, 1:], 97))
    hist_min = float(hist["close"].min())
    ax1.set_ylim(min(hist_min * 0.6, p03 * 0.6), p97 * 3.0)

    ax1.set_ylabel("Price (log scale, USD)")
    ax1.set_title(
        f"{symbol}  Probabilistic Price Forecast  ·  "
        f"OU half-life {ou_params['half_life']:.0f}d (95% CI: {ou_params['half_life']*0.5:.0f}–{ou_params['half_life']*1.5:.0f}d)  ·  "
        f"σ_daily={ou_params['sigma']:.3f}  ·  "
        f"as of {today.date()}\n"
        f"Shaded band = path uncertainty (1000 MC paths).  "
        f"Dotted lines = parameter uncertainty (θ ±50%).  "
        f"Do not conflate these two sources of uncertainty.",
        fontsize=9,
    )
    ax1.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax1.grid(True, which="both", alpha=0.4)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Z-score panel ─────────────────────────────────────────────────────────
    hist_z = features["zscore"].dropna()
    hist_z = hist_z[hist_z.index >= hist_start]
    ax2.plot(hist_z.index, hist_z.values, color=FG, lw=1.1, alpha=0.9)

    # Forward Z fan
    z_pcts = np.percentile(z_paths, [25, 50, 75], axis=0)
    ax2.fill_between(future_dates, z_pcts[0], z_pcts[2],
                     color=BLUE, alpha=0.20)
    ax2.plot(future_dates, z_pcts[1], color=BLUE, lw=1.0, ls="--", alpha=0.7)

    for thresh, color, ls in [(-1.5, GREEN, "--"), (1.5, ORANGE, "--"),
                               (-2.5, RED,   ":"),  (2.5, RED,   ":")]:
        ax2.axhline(thresh, color=color, lw=0.7, ls=ls, alpha=0.6)
        ax2.text(hist_z.index[0], thresh + 0.08, f"Z={thresh:+.1f}",
                 color=color, fontsize=7, alpha=0.8)

    ax2.axvline(today, color=GREY, lw=0.8, ls=":", alpha=0.7)
    ax2.scatter([today], [current_z], color=RED, s=50, zorder=10)
    ax2.axhline(0, color=GREY, lw=0.5, alpha=0.4)
    ax2.set_ylabel("Z-score")
    ax2.set_ylim(-5, 5)
    ax2.grid(True, alpha=0.35)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Probability table (text panel) ────────────────────────────────────────
    # Column layout: tighter spacing, larger font, colour-coded probabilities
    cols = [
        ("Horizon",          0.00,  8, YELLOW),
        ("25th pct",         0.14,  8, YELLOW),
        ("Median (50th)",    0.22,  8, YELLOW),
        ("75th pct",         0.33,  8, YELLOW),
        ("P(>1.5× now)",     0.44,  8, YELLOW),
        ("P(>2× now)",       0.55,  8, YELLOW),
        ("P(>3× now)",       0.65,  8, YELLOW),
        ("P(<−30%)",         0.76,  8, YELLOW),
    ]
    y_hdr = 0.92
    for hdr, x, fs, col in cols:
        ax3.text(x, y_hdr, hdr, color=col, fontsize=fs, fontweight="bold",
                 transform=ax3.transAxes)

    ax3.axhline(0.88, color=GRID, lw=0.5)

    for row_i, row in enumerate(prob_rows):
        y = y_hdr - 0.30 * (row_i + 1)
        # Colour-code probabilities: green if favourable, red if cautionary
        def prob_color(p: float, good_hi: bool = True) -> str:
            if good_hi:
                return GREEN if p >= 60 else ORANGE if p >= 35 else RED
            return RED if p >= 20 else ORANGE if p >= 10 else GREEN

        vals_cols = [
            (f"+{row['months']}mo  ({row['date']})",    0.00, FG),
            (f"${row['p25']:>9,.0f}",                   0.14, GREY),
            (f"${row['median']:>9,.0f}",                0.22, BLUE),
            (f"${row['p75']:>9,.0f}",                   0.33, GREY),
            (f"{row['pct_above_15x']:>5.0f}%",          0.44, prob_color(row['pct_above_15x'])),
            (f"{row['pct_above_2x']:>5.0f}%",           0.55, prob_color(row['pct_above_2x'])),
            (f"{row['pct_above_3x']:>5.0f}%",           0.65, prob_color(row['pct_above_3x'])),
            (f"{row['pct_below_07x']:>5.0f}%",          0.76, prob_color(row['pct_below_07x'], good_hi=False)),
        ]
        for txt, x, col in vals_cols:
            ax3.text(x, y, txt, color=col, fontsize=9, transform=ax3.transAxes,
                     fontweight="bold" if x == 0.22 else "normal")

    ax3.text(0.00, 0.02,
             "⚠  Statistical model only — not a financial forecast.  "
             "Z-score modelled as OU process (mean-reverting); power-law curve projected forward.",
             color=GREY, fontsize=7, transform=ax3.transAxes)

    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probabilistic BTC/ETH price forecast",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol",  default="BTC", choices=["BTC", "ETH"])
    p.add_argument("--horizon", type=int, default=18,
                   help="Forecast horizon in months")
    p.add_argument("--paths",   type=int, default=1000,
                   help="Number of Monte Carlo paths")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    symbol    = args.symbol
    yf_sym    = cfg.BTC_SYMBOL if symbol == "BTC" else cfg.ETH_SYMBOL
    genesis   = cfg.BTC_GENESIS if symbol == "BTC" else cfg.ETH_GENESIS

    print(f"\n  {symbol} Forecast  ({args.horizon}mo horizon, {args.paths} paths)")
    print(f"  {'─'*50}")

    print(f"  Fetching {yf_sym}...", end=" ", flush=True)
    df = fetch_ohlcv(yf_sym, start="2014-01-01")
    print(f"{len(df)} days")

    features = build_features(df, genesis_date=genesis, cfg=cfg)
    valid    = features.dropna(subset=["zscore", "log_deviation"])

    latest      = valid.iloc[-1]
    current_z   = float(latest["zscore"])
    current_p   = float(latest["close"])
    log_dev     = float(latest["log_deviation"])
    log_std     = abs(log_dev / current_z) if abs(current_z) > 0.01 else 0.25

    params = fit_power_law(features["close"], genesis)
    a, b   = params["a"], params["b"]

    # ── OU fit ────────────────────────────────────────────────────────────────
    ou = fit_ou(valid["zscore"])
    print(f"  OU fit:  half-life = {ou['half_life']:.0f} days  "
          f"·  daily σ = {ou['sigma']:.4f}  ·  φ = {ou['phi']:.4f}")
    print(f"  Current Z = {current_z:+.2f}  ·  log_std = {log_std:.3f}  "
          f"·  expected = ${float(latest['expected_price']):,.0f}")

    # ── Simulate ──────────────────────────────────────────────────────────────
    n_days       = args.horizon * 30
    today        = pd.Timestamp(valid.index[-1].date())
    future_dates = pd.date_range(start=today, periods=n_days + 1, freq="D")

    print(f"  Simulating {args.paths} paths × {n_days} days...", end=" ", flush=True)
    z_paths     = simulate_z(current_z, ou, n_days, n_paths=args.paths)
    price_paths = z_to_price(z_paths, future_dates, a, b, genesis, log_std)
    print("done")

    # ── Historical analogs ────────────────────────────────────────────────────
    analogs = find_analogs(features, z_lo=-3.5, z_hi=-1.8, fwd_days=n_days, n=4)
    print(f"  Found {len(analogs)} historical analogs:")
    for an in analogs:
        peak_gain = (an["prices"].max() / an["entry_price"] - 1) * 100
        print(f"    {an['entry_date'].strftime('%b %Y')}  "
              f"Z={an['entry_z']:+.2f}  entry=${an['entry_price']:,.0f}  "
              f"peak gain={peak_gain:+.0f}%  "
              f"({an['days']} days of data)")

    # ── Probability table ─────────────────────────────────────────────────────
    horizon_mos = [6, 12, 18] if args.horizon >= 18 else [6, 12]
    if args.horizon >= 24:
        horizon_mos.append(24)
    prob_rows = prob_table(price_paths, future_dates, current_p, horizon_mos)

    print(f"\n  Probability table:")
    print(f"  {'Horizon':<14} {'Median':>10} {'P(>1.5x)':>10} "
          f"{'P(>2x)':>8} {'P(>3x)':>8} {'P(<0.7x)':>9}")
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*9}")
    for row in prob_rows:
        print(f"  +{row['months']}mo ({row['date']})  "
              f"${row['median']:>8,.0f}  "
              f"{row['pct_above_15x']:>8.0f}%  "
              f"{row['pct_above_2x']:>6.0f}%  "
              f"{row['pct_above_3x']:>6.0f}%  "
              f"{row['pct_below_07x']:>7.0f}%")

    # ── Chart ─────────────────────────────────────────────────────────────────
    out_path = (ROOT / "reports" / "charts" /
                f"forecast_{symbol}_{date.today()}.png")
    make_chart(
        features, z_paths, price_paths, future_dates,
        analogs, params, genesis,
        current_p, current_z, log_std,
        ou, prob_rows, symbol, out_path,
    )


if __name__ == "__main__":
    main()
