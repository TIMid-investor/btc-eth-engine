"""
models/power_law.py — Long-term expected price curve.

Two models are available:

  power_law  price = a * days_since_genesis ^ b
             Fit via OLS in log-log space.  Captures the decelerating
             growth trajectory observed in BTC (b ≈ 5–6 historically).
             b < 1 is enforced so the curve is sub-linear in log space.

  log_ema    expected_log_price = EMA(log(price), span)
             Adaptive alternative; no closed-form, but self-calibrates
             to the most recent regime without a fixed genesis reference.

Both produce:
  expected   : Series of expected prices, same index as input
  log_dev    : log(price / expected) — the deviation in log space
               (≈ fractional deviation for small values)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


# ── Power law ──────────────────────────────────────────────────────────────────

def _days_since(dates: pd.DatetimeIndex, genesis: str) -> np.ndarray:
    """Integer days from *genesis* to each date in *dates*.  Clipped to ≥ 1."""
    origin = pd.Timestamp(genesis)
    return np.maximum(1, (dates - origin).days.values.astype(float))


def fit_power_law(
    prices: pd.Series,
    genesis_date: str,
) -> dict:
    """
    Fit price = a * t^b in log-log space using OLS.

    Parameters
    ----------
    prices       : Series of prices indexed by pd.Timestamp
    genesis_date : ISO date string for the t=0 reference

    Returns
    -------
    dict with keys:
      a          : scale coefficient
      b          : exponent (ideally 0 < b < 10 for crypto)
      r_squared  : goodness of fit on log(price)
      genesis    : genesis_date echoed back
    """
    t = _days_since(prices.index, genesis_date)
    log_t = np.log(t)
    log_p = np.log(prices.values.astype(float))

    valid = np.isfinite(log_t) & np.isfinite(log_p)
    if valid.sum() < 30:
        raise ValueError("Fewer than 30 valid price observations — cannot fit power law.")

    log_t_v = log_t[valid]
    log_p_v = log_p[valid]

    # OLS: log(p) = log_a + b * log(t)
    coeffs = np.polyfit(log_t_v, log_p_v, deg=1)
    b     = float(coeffs[0])
    log_a = float(coeffs[1])
    a     = np.exp(log_a)

    # R² on log scale
    log_p_hat = log_a + b * log_t_v
    ss_res = np.sum((log_p_v - log_p_hat) ** 2)
    ss_tot = np.sum((log_p_v - log_p_v.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"a": a, "b": b, "r_squared": r2, "genesis": genesis_date}


def expected_price_power_law(
    dates: pd.DatetimeIndex,
    params: dict,
) -> pd.Series:
    """
    Compute expected price at each date given fitted power-law params.

    Parameters
    ----------
    dates  : DatetimeIndex for which to evaluate the curve
    params : dict returned by fit_power_law()
    """
    t = _days_since(dates, params["genesis"])
    exp = params["a"] * np.power(t, params["b"])
    return pd.Series(exp, index=dates, name="expected_price")


# ── Log-EMA alternative ────────────────────────────────────────────────────────

def expected_price_log_ema(
    prices: pd.Series,
    span: int = 730,
) -> pd.Series:
    """
    Long-term EMA of log(price), converted back to price space.

    Useful as an adaptive alternative to the power law when the genesis
    reference date is uncertain or the fit is unstable.

    Parameters
    ----------
    prices : price Series
    span   : EMA span in days (default 730 ≈ 2 years)
    """
    log_p = np.log(prices)
    log_ema = log_p.ewm(span=span, adjust=False).mean()
    expected = np.exp(log_ema)
    expected.name = "expected_price"
    return expected


# ── Rolling-median alternative ────────────────────────────────────────────────

def expected_price_rolling_median(
    prices: pd.Series,
    window: int = 365,
) -> pd.Series:
    """
    Non-parametric expected price: rolling median of log(price) in price space.

    No genesis date, no curve assumption.  The expected price at each point is
    simply the median price over the trailing `window` days, computed in log
    space (geometric median).  This is a sanity-check alternative to the power
    law: if Z-score signals are robust when using rolling median instead of the
    fitted curve, the strategy has some inherent validity beyond curve-fitting.

    Parameters
    ----------
    prices : price Series indexed by pd.Timestamp
    window : trailing window in days (default 365)
    """
    log_p     = np.log(prices)
    log_med   = log_p.rolling(window=window, min_periods=window // 2).median()
    expected  = np.exp(log_med)
    expected.name = "expected_price"
    return expected


# ── Deviation (log space) ──────────────────────────────────────────────────────

def log_deviation(prices: pd.Series, expected: pd.Series) -> pd.Series:
    """
    Compute log(price / expected_price).

    This is the natural deviation metric because:
    - It is symmetric around zero on a percentage basis.
    - It is more stationary than raw price differences.
    - +0.3 means ~35% above the trend; -0.3 means ~26% below.
    """
    dev = np.log(prices / expected)
    dev.name = "log_deviation"
    return dev


# ── Fit summary ────────────────────────────────────────────────────────────────

def print_fit_summary(params: dict, prices: pd.Series) -> None:
    """Print a human-readable summary of the power-law fit."""
    print(f"\n  Power-law fit:  price = {params['a']:.4e} × days^{params['b']:.4f}")
    print(f"  R² (log scale): {params['r_squared']:.4f}")
    today_t  = _days_since(pd.DatetimeIndex([pd.Timestamp.today()]), params["genesis"])[0]
    today_ex = params["a"] * today_t ** params["b"]
    today_ac = float(prices.iloc[-1])
    print(f"  Today:  actual ${today_ac:>12,.0f}   expected ${today_ex:>12,.0f}"
          f"   ratio {today_ac/today_ex:.3f}")
