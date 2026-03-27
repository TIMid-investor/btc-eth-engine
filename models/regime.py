"""
models/regime.py — Market regime classification and threshold adjustment.

Two layers of regime awareness:

1.  Halving cycle (BTC only)
    Tags each date as one of four phases relative to the nearest halving:
      PRE_HALVING      — within 6 months before a halving
      POST_EARLY       — 0–12 months after a halving  (historically strongest bull)
      POST_LATE        — 12–24 months after a halving (trend still up, momentum fading)
      LATE_CYCLE       — 24+ months after a halving  (overextension / correction risk)

2.  Price regime (BTC and ETH)
    Classifies each date using price relative to its 200-day SMA and drawdown from ATH:
      BULL             — price above 200d SMA by ≥ 5% and drawdown > -30%
      BEAR             — price below 200d SMA by ≥ 10% or drawdown < -50%
      ACCUMULATION     — recovering from bear (price crossing 200d SMA from below)
      NEUTRAL          — everything else

Combined into a single REGIME string: e.g. "POST_EARLY|BULL"

Threshold adjustments:
    Regime-specific multipliers are applied to the base Z-score thresholds so
    the model requires stronger signals (higher |Z|) in risky periods and allows
    weaker signals in historically productive regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── BTC halving dates ─────────────────────────────────────────────────────────

BTC_HALVINGS = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-19"),
    # Next expected ~2028-04 (approximated)
    pd.Timestamp("2028-04-15"),
]

# Days since last halving → phase label
HALVING_PHASES = [
    (0,   180,  "PRE_HALVING"),   # actually days BEFORE next halving; see logic below
    (0,   365,  "POST_EARLY"),
    (365, 730,  "POST_LATE"),
    (730, 9999, "LATE_CYCLE"),
]


def halving_phase(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Return the halving-cycle phase for each date.

    Phases:
      PRE_HALVING  — within 180 days before the NEXT halving
      POST_EARLY   — 0–365 days after the most recent halving
      POST_LATE    — 365–730 days after the most recent halving
      LATE_CYCLE   — 730+ days after the most recent halving

    Returns Series of string labels indexed by `dates`.
    """
    phases = []
    halvings_sorted = sorted(BTC_HALVINGS)

    for dt in dates:
        # Find the most recent past halving
        past = [h for h in halvings_sorted if h <= dt]
        future = [h for h in halvings_sorted if h > dt]

        if not past:
            phases.append("PRE_HALVING")
            continue

        last_halving = past[-1]
        days_since = (dt - last_halving).days

        # Check if within 180 days of the NEXT halving
        if future and (future[0] - dt).days <= 180:
            phases.append("PRE_HALVING")
        elif days_since <= 365:
            phases.append("POST_EARLY")
        elif days_since <= 730:
            phases.append("POST_LATE")
        else:
            phases.append("LATE_CYCLE")

    return pd.Series(phases, index=dates, name="halving_phase")


# ── Price regime ───────────────────────────────────────────────────────────────

def price_regime(prices: pd.Series, sma_window: int = 200) -> pd.Series:
    """
    Classify each date into BULL / BEAR / ACCUMULATION / NEUTRAL based on:
      - Price vs its 200-day SMA
      - Drawdown from rolling 365-day high

    Returns Series of string labels, same index as `prices`.
    """
    sma    = prices.rolling(window=sma_window, min_periods=sma_window // 2).mean()
    rel    = prices / sma - 1.0           # positive → above SMA
    ath    = prices.rolling(window=365, min_periods=30).max()
    dd     = prices / ath - 1.0           # 0 = at ATH, -0.5 = 50% below ATH

    # Crossing 200d SMA from below (accumulation signal)
    crossing_up = (rel > 0) & (rel.shift(1) <= 0)

    regimes = []
    accumulation_window = 0

    for i in range(len(prices)):
        r   = float(rel.iloc[i])
        d   = float(dd.iloc[i])
        cu  = bool(crossing_up.iloc[i])

        if cu:
            accumulation_window = 60   # label next 60 days as accumulation

        if np.isnan(r) or np.isnan(d):
            regimes.append("UNKNOWN")
        elif d < -0.50 or r < -0.10:
            regimes.append("BEAR")
            accumulation_window = 0
        elif accumulation_window > 0:
            regimes.append("ACCUMULATION")
            accumulation_window -= 1
        elif r >= 0.05 and d > -0.30:
            regimes.append("BULL")
        else:
            regimes.append("NEUTRAL")

    return pd.Series(regimes, index=prices.index, name="price_regime")


# ── Combined regime ────────────────────────────────────────────────────────────

def classify_regime(
    prices: pd.Series,
    include_halving: bool = True,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns: halving_phase, price_regime, regime.

    `regime` is the combined label, e.g. "POST_EARLY|BULL".
    If `include_halving` is False (e.g. for ETH), only price_regime is used.
    """
    pr = price_regime(prices)

    if include_halving:
        hp = halving_phase(prices.index)
        combined = hp.str.cat(pr, sep="|")
    else:
        hp = pd.Series("N/A", index=prices.index, name="halving_phase")
        combined = pr.copy()

    combined.name = "regime"
    return pd.DataFrame({"halving_phase": hp, "price_regime": pr, "regime": combined})


# ── Threshold adjustments ─────────────────────────────────────────────────────

# Maps (halving_phase, price_regime) → buy_threshold multiplier.
# > 1.0 means require a STRONGER signal (more oversold) to enter.
# < 1.0 means allow entry at a WEAKER signal.
_THRESHOLD_MULTIPLIERS: dict[tuple[str, str], float] = {
    # Post-halving early + bull: price tends to keep running up; be quicker to buy dips
    ("POST_EARLY",  "BULL"):         0.80,
    ("POST_EARLY",  "NEUTRAL"):      0.90,
    ("POST_EARLY",  "ACCUMULATION"): 0.85,

    # Post-halving late: still productive but more cautious
    ("POST_LATE",   "BULL"):         0.90,
    ("POST_LATE",   "NEUTRAL"):      1.00,

    # Late cycle: higher conviction required — overextension risk
    ("LATE_CYCLE",  "BULL"):         1.10,
    ("LATE_CYCLE",  "NEUTRAL"):      1.15,
    ("LATE_CYCLE",  "BEAR"):         1.30,   # avoid longs in late-cycle bear

    # Pre-halving: neutral; uncertainty high
    ("PRE_HALVING", "BULL"):         1.00,
    ("PRE_HALVING", "NEUTRAL"):      1.05,
    ("PRE_HALVING", "BEAR"):         1.20,

    # Bear + any phase: require strong oversold signal
    ("POST_EARLY",  "BEAR"):         1.25,
    ("POST_LATE",   "BEAR"):         1.30,
}

_DEFAULT_MULTIPLIER = 1.00


def regime_threshold_multiplier(halving_ph: str, price_reg: str) -> float:
    """Return the Z-score threshold multiplier for a given regime combination."""
    return _THRESHOLD_MULTIPLIERS.get((halving_ph, price_reg), _DEFAULT_MULTIPLIER)


def build_regime_frame(
    prices: pd.Series,
    base_buy_threshold: float,
    include_halving: bool = True,
) -> pd.DataFrame:
    """
    Build a DataFrame with regime columns and adjusted buy thresholds.

    Columns:
      halving_phase, price_regime, regime,
      threshold_mult, adjusted_buy_threshold
    """
    regime_df = classify_regime(prices, include_halving=include_halving)

    mults = [
        regime_threshold_multiplier(str(hp), str(pr))
        for hp, pr in zip(regime_df["halving_phase"], regime_df["price_regime"])
    ]
    regime_df["threshold_mult"]        = mults
    regime_df["adjusted_buy_threshold"] = base_buy_threshold * np.array(mults)

    return regime_df


# ── Regime-aware signal gating ────────────────────────────────────────────────

def apply_regime_to_target(
    target_position: pd.Series,
    zscore: pd.Series,
    regime_df: pd.DataFrame,
    base_buy_threshold: float,
) -> pd.Series:
    """
    Re-evaluate the target position using regime-adjusted thresholds.

    For each row, if |zscore| does not exceed the regime-adjusted threshold,
    the target position is set to 0 (no trade).

    Returns adjusted target_position Series.
    """
    adjusted = target_position.copy()
    adj_thresh = regime_df["adjusted_buy_threshold"].reindex(target_position.index)

    # Zero out buy signals that don't meet the regime-adjusted threshold
    below_adj_thresh = zscore.abs() < adj_thresh
    adjusted[below_adj_thresh & (target_position > 0)] = 0.0

    return adjusted
