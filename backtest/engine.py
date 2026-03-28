"""
backtest/engine.py — Core simulation engine.

Pipeline
--------
1.  build_features(df, params, cfg) → enriched DataFrame with:
      expected_price, log_deviation, zscore
      trend, volume_ok, macro_ok
      target_position (continuous, signed)

    build_features_walk_forward(df, genesis, cfg, refit_months) →
      Same output, but power-law is refit quarterly using only past data.
      Eliminates look-ahead bias in the curve parameters.

2.  run_backtest(features, cfg) → (equity_curve, trades_log)
      Simulates daily mark-to-market with fees and slippage.
      Rebalances only when position drifts outside REBALANCE_BAND.

Position sizing
---------------
  Long signal  (Z < -BUY_THRESHOLD):
    size = clip( (|Z| - threshold) / (ZSCORE_SCALE - threshold), 0, 1 ) × MAX_POSITION

  Short signal (Z > SELL_THRESHOLD, LONG_ONLY=False):
    size = -clip( (Z - threshold) / (ZSCORE_SCALE - threshold), 0, 1 ) × MAX_POSITION

  Exit zone   (|Z| < EXIT_THRESHOLD): target = 0 (full exit)
  Between thresholds: hold current position (avoid whipsawing around the band)

Filters applied before computing target_position:
  - trend_filter  : long only allowed when EMA slope positive (if USE_TREND_FILTER)
  - volume_filter : signal requires volume above minimum ratio (if USE_VOLUME_FILTER)
  - macro_filter  : no new entries during severe drawdowns (if USE_MACRO_FILTER)

Fees and slippage are applied on each rebalance as a fraction of the dollar
amount traded (not the portfolio value).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as _default_cfg
from models.power_law import (
    fit_power_law,
    expected_price_power_law,
    expected_price_log_ema,
    log_deviation,
)
from models.zscore import rolling_zscore
from models.filters import build_filter_frame
from models.regime import build_regime_frame, apply_regime_to_target


# ── Feature builder ────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    genesis_date: str,
    cfg=_default_cfg,
    use_regime: bool = False,
) -> pd.DataFrame:
    """
    Enrich the raw OHLCV DataFrame with all model features.

    Parameters
    ----------
    df           : DataFrame with columns open/high/low/close/volume, date index
    genesis_date : ISO date for the power-law origin (BTC_GENESIS or ETH_GENESIS)
    cfg          : config module (defaults to config.py)

    Returns
    -------
    DataFrame with additional columns:
      expected_price, log_deviation, zscore,
      trend, volume_ok, macro_ok, raw_signal, target_position
    """
    out = df.copy()

    # ── Expected price ────────────────────────────────────────────────────────
    if cfg.CURVE_MODEL == "power_law":
        params = fit_power_law(out["close"], genesis_date)
        out["expected_price"] = expected_price_power_law(out.index, params)
        out["curve_params_b"] = params["b"]
    else:
        out["expected_price"] = expected_price_log_ema(out["close"], span=cfg.LOG_EMA_SPAN)
        out["curve_params_b"] = float("nan")

    out["log_deviation"] = log_deviation(out["close"], out["expected_price"])

    # ── Z-score ───────────────────────────────────────────────────────────────
    out["zscore"] = rolling_zscore(
        out["log_deviation"],
        window=cfg.ZSCORE_WINDOW,
        min_periods=cfg.ZSCORE_MIN_PERIODS,
    )

    # ── Filters ───────────────────────────────────────────────────────────────
    filters = build_filter_frame(out["close"], out["volume"], cfg)
    out["trend"]      = filters["trend"]
    out["trend_mult"] = filters["trend_mult"]
    out["volume_ok"]  = filters["volume_ok"]
    out["macro_ok"]   = filters["macro_ok"]

    # ── Target position (continuous, range [-1, 1]) ───────────────────────────
    out["target_position"] = _compute_target_position(out, cfg)

    # ── Regime awareness (optional) ───────────────────────────────────────────
    if use_regime:
        include_halving = (genesis_date == "2009-01-03")  # BTC only
        regime_df = build_regime_frame(
            out["close"],
            base_buy_threshold=cfg.BUY_THRESHOLD,
            include_halving=include_halving,
        )
        out["halving_phase"] = regime_df["halving_phase"]
        out["price_regime"]  = regime_df["price_regime"]
        out["regime"]        = regime_df["regime"]
        out["threshold_mult"] = regime_df["threshold_mult"]
        out["target_position"] = apply_regime_to_target(
            out["target_position"], out["zscore"],
            regime_df, cfg.BUY_THRESHOLD,
        )

    return out


def _compute_target_position(df: pd.DataFrame, cfg) -> pd.Series:
    """
    Vectorised target position calculation.

    Position is continuous and scales linearly from 0 at threshold to
    MAX_POSITION at ZSCORE_SCALE.  The current position is held when Z
    is between the entry threshold and the exit threshold (dead zone).
    Filters can block or reduce the signal.
    """
    z = df["zscore"]
    threshold_buy  = cfg.BUY_THRESHOLD
    threshold_sell = cfg.SELL_THRESHOLD
    exit_z         = cfg.EXIT_THRESHOLD
    scale          = cfg.ZSCORE_SCALE
    max_pos        = cfg.MAX_POSITION
    long_only      = cfg.LONG_ONLY

    # Raw continuous signal (ignoring filters)
    raw = pd.Series(0.0, index=z.index)

    # Long zone: Z < -threshold_buy
    buy_mask = z < -threshold_buy
    buy_size = ((z.abs() - threshold_buy) / (scale - threshold_buy)).clip(0, 1) * max_pos
    raw[buy_mask] = buy_size[buy_mask]

    if not long_only:
        # Short zone: Z > threshold_sell
        sell_mask = z > threshold_sell
        sell_size = ((z - threshold_sell) / (scale - threshold_sell)).clip(0, 1) * max_pos
        raw[sell_mask] = -sell_size[sell_mask]

    # Exit zone: |Z| < exit_z → target = 0
    exit_mask = z.abs() < exit_z
    raw[exit_mask] = 0.0

    # NaN Z-score → no position
    raw[z.isna()] = 0.0

    # Apply trend filter: scale position by continuous slope multiplier.
    # trend_mult ≈ 1.0 in strong uptrend, ~0.82 in flat, ~0.07 in strong downtrend.
    # For short positions the multiplier is inverted (1 - mult) so a downtrend
    # amplifies shorts rather than suppressing them.
    if cfg.USE_TREND_FILTER:
        mult = df["trend_mult"].fillna(0.5)
        longs  = raw > 0
        shorts = raw < 0
        raw[longs]  = raw[longs]  * mult[longs]
        if not long_only:
            raw[shorts] = raw[shorts] * (1.0 - mult[shorts])

    # Apply volume filter: zero out signal on low-volume days
    if cfg.USE_VOLUME_FILTER:
        low_vol = ~df["volume_ok"].fillna(True)
        raw[low_vol] = 0.0

    # Apply macro filter: zero out signal during shock drawdowns
    if cfg.USE_MACRO_FILTER:
        shock = ~df["macro_ok"].fillna(True)
        raw[shock] = 0.0

    raw.name = "target_position"
    return raw


# ── Backtest simulator ────────────────────────────────────────────────────────

def run_backtest(
    features: pd.DataFrame,
    cfg=_default_cfg,
    start_date: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Simulate the strategy on the feature-enriched DataFrame.

    Parameters
    ----------
    features   : DataFrame from build_features()
    cfg        : config module
    start_date : ISO date to begin simulation (allows burn-in for Z-score)

    Returns
    -------
    equity_curve : pd.Series (daily portfolio value in $)
    trades       : pd.DataFrame with one row per closed trade
    """
    df = features.copy()
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    df = df.dropna(subset=["zscore"])

    if df.empty:
        raise ValueError("No rows with valid Z-scores after filtering — increase history or reduce ZSCORE_MIN_PERIODS.")

    capital   = cfg.INITIAL_CAPITAL
    position  = 0.0           # current fraction of capital in crypto
    entry_price = None
    entry_date  = None
    entry_capital = None

    equity_values = []
    trade_records = []

    prev_close = None

    for i, (date, row) in enumerate(df.iterrows()):
        close = float(row["close"])

        # ── Daily mark-to-market ──────────────────────────────────────────────
        if prev_close is not None and prev_close > 0 and position != 0.0:
            daily_ret = close / prev_close - 1.0
            capital  += position * capital * daily_ret

        prev_close = close

        # ── Rebalance ─────────────────────────────────────────────────────────
        target = float(row["target_position"])

        # Hold-through-cycle mode: override the vectorised continuous target.
        # Once in a long, stay fully invested until Z turns overbought.
        # Entries are still gated by the vectorised signal (filters apply).
        if getattr(cfg, "HOLD_THROUGH_CYCLE", False):
            z_now = float(row["zscore"]) if not np.isnan(row["zscore"]) else 0.0
            if position > 0:
                # In a long — hold unless Z has crossed above sell threshold
                if z_now > cfg.SELL_THRESHOLD:
                    target = 0.0          # overbought → exit
                else:
                    target = position     # hold at current size
            elif target > 0:
                # Not yet in — take a buy signal at full size
                target = cfg.MAX_POSITION

        delta  = target - position

        if abs(delta) >= cfg.REBALANCE_BAND:
            trade_dollars = abs(delta) * capital
            cost = trade_dollars * (cfg.FEE_RATE + cfg.SLIPPAGE)
            capital -= cost

            # Log trade entry/exit
            if position == 0.0 and target != 0.0:
                # Opening a new trade
                entry_price   = close
                entry_date    = date
                entry_capital = capital
                direction     = "LONG" if target > 0 else "SHORT"

            elif target == 0.0 and position != 0.0 and entry_date is not None:
                # Closing a trade
                exit_pnl = (capital - entry_capital) if entry_capital else float("nan")
                trade_records.append({
                    "entry_date":  entry_date,
                    "exit_date":   date,
                    "direction":   "LONG" if position > 0 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price":  close,
                    "entry_z":     float(df.at[entry_date, "zscore"]) if entry_date in df.index else float("nan"),
                    "exit_z":      float(row["zscore"]),
                    "pnl":         exit_pnl,
                    "pnl_pct":     exit_pnl / entry_capital if entry_capital else float("nan"),
                })
                entry_price   = None
                entry_date    = None
                entry_capital = None

            position = target

        equity_values.append({"date": date, "capital": capital})

    equity_curve = (
        pd.DataFrame(equity_values)
        .set_index("date")["capital"]
    )

    trades = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
        columns=["entry_date", "exit_date", "direction",
                 "entry_price", "exit_price", "entry_z", "exit_z",
                 "pnl", "pnl_pct"]
    )

    return equity_curve, trades


# ── Walk-forward feature builder ─────────────────────────────────────────────

def build_features_walk_forward(
    df: pd.DataFrame,
    genesis_date: str,
    cfg=_default_cfg,
    refit_months: int = 3,
    min_fit_years: float = 1.5,
) -> pd.DataFrame:
    """
    Build features with a walk-forward power-law curve — no look-ahead bias.

    At each refit date the curve is fit using only data available up to that
    point. The resulting expected prices are then used for the following
    `refit_months` months. The Z-score still uses a trailing rolling window.

    Parameters
    ----------
    df             : raw OHLCV DataFrame
    genesis_date   : ISO date for the power-law time origin
    cfg            : config module
    refit_months   : how often to refit the curve (default: quarterly)
    min_fit_years  : minimum history required before first fit

    Returns
    -------
    DataFrame with the same columns as build_features(), plus:
      curve_fit_date   : the date of the power-law fit used for each row
      walk_forward     : True (marks this as a walk-forward result)
    """
    out = df.copy()
    dates = out.index
    min_fit_rows = int(min_fit_years * 365)

    # Generate refit dates (every `refit_months` months after min_fit_rows)
    refit_dates: list[pd.Timestamp] = []
    step = pd.DateOffset(months=refit_months)
    first_eligible = dates[min_fit_rows] if len(dates) > min_fit_rows else dates[-1]
    current = first_eligible
    while current <= dates[-1]:
        # Find the nearest actual trading day at or before `current`
        valid = dates[dates <= current]
        if len(valid):
            refit_dates.append(valid[-1])
        current += step

    if not refit_dates:
        raise ValueError(
            f"Not enough history for walk-forward fit. "
            f"Need at least {min_fit_years:.1f} years of data."
        )

    # For each row, find its refit date and apply the corresponding fit
    # We'll build the expected_price series segment by segment
    expected_prices = pd.Series(np.nan, index=dates)
    fit_date_labels = pd.Series("", index=dates)

    # Build list of (fit_date, apply_start, apply_end) segments
    segments = []
    for i, fit_date in enumerate(refit_dates):
        apply_start = fit_date
        apply_end   = refit_dates[i + 1] if i + 1 < len(refit_dates) else dates[-1]
        segments.append((fit_date, apply_start, apply_end))

    from models.power_law import (
        fit_power_law as _fit, expected_price_power_law as _exp,
        expected_price_log_ema as _ema,
    )

    for fit_date, apply_start, apply_end in segments:
        past_prices = out.loc[out.index <= fit_date, "close"]
        if len(past_prices) < 60:
            continue
        try:
            if cfg.CURVE_MODEL == "power_law":
                params = _fit(past_prices, genesis_date)
                segment_idx = dates[(dates >= apply_start) & (dates <= apply_end)]
                exp = _exp(segment_idx, params)
            else:
                # For log_ema: recompute on past data only, take last value as anchor
                past_exp = _ema(past_prices, span=cfg.LOG_EMA_SPAN)
                segment_idx = dates[(dates >= apply_start) & (dates <= apply_end)]
                # Extend the EMA forward using the last fitted value (frozen)
                last_val = float(past_exp.iloc[-1])
                exp = pd.Series(last_val, index=segment_idx)

            expected_prices.loc[segment_idx] = exp.values
            fit_date_labels.loc[segment_idx] = str(fit_date.date())
        except Exception:
            continue

    out["expected_price"] = expected_prices
    out["curve_fit_date"] = fit_date_labels
    out["curve_params_b"] = float("nan")

    # Rows before first fit → NaN (burn-in)
    out.loc[out["expected_price"].isna(), "expected_price"] = np.nan

    # Drop rows with no expected price
    out = out.dropna(subset=["expected_price"])

    # Log deviation and Z-score (same as build_features)
    out["log_deviation"] = log_deviation(out["close"], out["expected_price"])
    out["zscore"] = rolling_zscore(
        out["log_deviation"],
        window=cfg.ZSCORE_WINDOW,
        min_periods=cfg.ZSCORE_MIN_PERIODS,
    )

    # Filters and target position
    filters = build_filter_frame(out["close"], out["volume"], cfg)
    out["trend"]      = filters["trend"]
    out["trend_mult"] = filters["trend_mult"]
    out["volume_ok"]  = filters["volume_ok"]
    out["macro_ok"]   = filters["macro_ok"]
    out["target_position"] = _compute_target_position(out, cfg)
    out["walk_forward"] = True

    return out


# ── Buy-and-hold benchmark ────────────────────────────────────────────────────

def buy_and_hold(features: pd.DataFrame, cfg=_default_cfg, start_date: str | None = None) -> pd.Series:
    """
    Simple buy-and-hold benchmark starting with INITIAL_CAPITAL.
    Full investment on day 1; no fees.
    """
    df = features.copy()
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    df = df.dropna(subset=["close"])
    if df.empty:
        return pd.Series(dtype=float)
    norm = df["close"] / float(df["close"].iloc[0])
    return (norm * cfg.INITIAL_CAPITAL).rename("buy_and_hold")
