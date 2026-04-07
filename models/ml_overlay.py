"""
models/ml_overlay.py — Walk-forward ML signal classifier.

The ML overlay sits on top of the rule-based signal. At each trade entry
the model predicts the probability that the trade will be profitable. Only
trades above a confidence threshold are actually executed.

Features (computed per signal day):
  zscore          — current Z-score (the primary signal strength)
  zscore_velocity — rate of change of Z-score (dZ/dt, 5-day)
  zscore_accel    — acceleration of Z-score (d²Z/dt², 5-day)
  trend_slope     — normalised 20-wk EMA slope (momentum of the trend)
  volume_ratio    — today's volume / 30-day avg volume
  log_deviation   — log(price/expected), the raw deviation
  days_in_zone    — consecutive days Z has been in the buy zone
  price_regime    — encoded: BULL=1, NEUTRAL=0, ACCUMULATION=0.5, BEAR=-1

Training:
  Walk-forward: train on all CLOSED trades up to the current entry date.
  Minimum 20 trades required before the classifier is activated.
  Refit classifier every N new trades (default: 10).

Classifier:
  Logistic regression (fast, interpretable, regularised).
  Optionally XGBoost if available.

Label:
  1 = winning trade (pnl > 0 after fees)
  0 = losing trade

Output:
  signal_confidence [0, 1] per entry signal.
  Trades below CONFIDENCE_THRESHOLD are suppressed.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ── Feature engineering ───────────────────────────────────────────────────────

REGIME_ENCODING = {
    "BULL":         1.0,
    "ACCUMULATION": 0.5,
    "NEUTRAL":      0.0,
    "BEAR":        -1.0,
    "UNKNOWN":      0.0,
}


def build_ml_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ML feature matrix from the enriched features DataFrame.

    Returns DataFrame with one row per day (same index), containing:
      zscore, zscore_vel5, zscore_accel5, trend_slope,
      volume_ratio, log_deviation, days_in_zone, price_regime_enc
    """
    z  = features["zscore"]
    lv = features.get("volume", None)
    ld = features.get("log_deviation", None)
    close = features["close"]

    # Z-score velocity and acceleration (5-day finite difference)
    z_vel5   = z.diff(5) / 5.0
    z_accel5 = z_vel5.diff(5) / 5.0

    # EMA slope — normalised by price level
    ema_140 = close.ewm(span=140, adjust=False).mean()
    ema_slope = ema_140.diff(5) / ema_140.shift(5)  # fractional slope

    # Volume ratio
    if lv is not None:
        vol_avg = lv.rolling(30, min_periods=5).mean()
        vol_ratio = (lv / vol_avg.replace(0, np.nan)).clip(0, 5)
    else:
        vol_ratio = pd.Series(1.0, index=features.index)

    # Consecutive days in the buy zone (Z < -threshold)
    # Approximated with a sign-change counter
    in_zone = (z < -1.0).astype(int)
    days_in_zone = in_zone.groupby((in_zone != in_zone.shift()).cumsum()).cumcount() + 1
    days_in_zone = days_in_zone.where(in_zone == 1, 0)

    # Price regime encoding
    if "price_regime" in features.columns:
        regime_enc = features["price_regime"].map(REGIME_ENCODING).fillna(0.0)
    else:
        regime_enc = pd.Series(0.0, index=features.index)

    ml_df = pd.DataFrame({
        "zscore":          z,
        "zscore_vel5":     z_vel5,
        "zscore_accel5":   z_accel5,
        "trend_slope":     ema_slope,
        "volume_ratio":    vol_ratio,
        "log_deviation":   ld if ld is not None else pd.Series(np.nan, index=features.index),
        "days_in_zone":    days_in_zone,
        "price_regime_enc": regime_enc,
    }, index=features.index)

    return ml_df


FEATURE_COLS = [
    "zscore", "zscore_vel5", "zscore_accel5",
    "trend_slope", "volume_ratio", "log_deviation",
    "days_in_zone", "price_regime_enc",
]


# ── Walk-forward classifier ───────────────────────────────────────────────────

class WalkForwardClassifier:
    """
    Maintains a rolling logistic regression model that is refit
    every `refit_every` new closed trades.

    Usage:
        clf = WalkForwardClassifier()
        for entry_date, pnl, features_row in trade_history:
            clf.add_trade(entry_date, features_row, label=(pnl > 0))
            confidence = clf.predict(current_features_row)
    """

    def __init__(
        self,
        min_trades: int = 20,
        refit_every: int = 10,
        confidence_threshold: float = 0.55,
        max_iter: int = 1000,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for the ML overlay. "
                              "Install with: pip install scikit-learn")
        self.min_trades           = min_trades
        self.refit_every          = refit_every
        self.confidence_threshold = confidence_threshold
        self.max_iter             = max_iter

        self._X: list[list[float]] = []
        self._y: list[int]         = []
        self._trade_count          = 0
        self._last_refit_at        = 0
        self._pipeline: Optional[Pipeline] = None
        self._feature_importances: dict[str, float] = {}

    def add_trade(
        self,
        features_row: pd.Series,
        label: bool,
    ) -> None:
        """Add a completed trade to the training set."""
        x = [float(features_row.get(c, 0.0)) for c in FEATURE_COLS]
        if any(np.isnan(v) for v in x):
            return  # skip rows with NaN features
        self._X.append(x)
        self._y.append(int(label))
        self._trade_count += 1

        # Refit when we have enough trades and it's been long enough
        if (self._trade_count >= self.min_trades and
                self._trade_count - self._last_refit_at >= self.refit_every):
            self._refit()

    def _refit(self) -> None:
        """Train the logistic regression on all accumulated trades."""
        X = np.array(self._X)
        y = np.array(self._y)

        # Need at least both classes present
        if len(np.unique(y)) < 2:
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(
                    C=1.0, max_iter=self.max_iter,
                    class_weight="balanced", random_state=42,
                )),
            ])
            pipe.fit(X, y)

        self._pipeline = pipe
        self._last_refit_at = self._trade_count

        # Store feature importances (logistic regression coefficients)
        coefs = pipe["clf"].coef_[0]
        self._feature_importances = dict(zip(FEATURE_COLS, np.abs(coefs)))

    def predict(self, features_row: pd.Series) -> float:
        """
        Return P(trade wins) ∈ [0, 1].
        Returns 0.5 (uncertain) if model is not yet trained.
        """
        if self._pipeline is None:
            return 0.5
        x = np.array([[float(features_row.get(c, 0.0)) for c in FEATURE_COLS]])
        if np.any(np.isnan(x)):
            return 0.5
        try:
            prob = float(self._pipeline.predict_proba(x)[0][1])
        except Exception:
            return 0.5
        return prob

    def is_ready(self) -> bool:
        return self._pipeline is not None

    @property
    def n_trades(self) -> int:
        return self._trade_count

    @property
    def feature_importances(self) -> dict[str, float]:
        return self._feature_importances


# ── ML-augmented backtest ─────────────────────────────────────────────────────

def run_backtest_with_ml(
    features: pd.DataFrame,
    cfg,
    start_date: str | None = None,
    confidence_threshold: float = 0.55,
    min_trades_to_activate: int = 20,
    refit_every: int = 10,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Run the backtest with the ML overlay gating trade entries.

    Returns
    -------
    equity_curve   : daily portfolio value
    trades         : closed trades log (with confidence column)
    confidence_log : daily confidence scores for all signal days
    """
    df = features.copy()
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    df = df.dropna(subset=["zscore"])

    if df.empty:
        raise ValueError("No valid Z-score rows after filtering.")

    # Build ML feature matrix
    ml_feats = build_ml_features(features)
    ml_feats = ml_feats.reindex(df.index)

    classifier = WalkForwardClassifier(
        min_trades=min_trades_to_activate,
        refit_every=refit_every,
        confidence_threshold=confidence_threshold,
    )

    capital    = cfg.INITIAL_CAPITAL
    position   = 0.0
    entry_price   = None
    entry_date    = None
    entry_capital = None
    entry_features = None

    equity_values: list[dict]    = []
    trade_records: list[dict]    = []
    confidence_records: list[dict] = []

    prev_close = None

    for date, row in df.iterrows():
        close = float(row["close"])

        # Mark-to-market
        if prev_close is not None and prev_close > 0 and position != 0.0:
            daily_ret = close / prev_close - 1.0
            capital  += position * capital * daily_ret
        prev_close = close

        target = float(row["target_position"])
        ml_row = ml_feats.loc[date] if date in ml_feats.index else pd.Series()

        # Gate the signal with ML confidence
        confidence = 0.5
        if target > 0 and position == 0.0:
            confidence = classifier.predict(ml_row)
            confidence_records.append({"date": date, "confidence": confidence,
                                        "zscore": float(row["zscore"]),
                                        "signal": target > 0})
            if classifier.is_ready() and confidence < confidence_threshold:
                target = 0.0  # ML suppressed this trade

        delta = target - position
        if abs(delta) >= cfg.REBALANCE_BAND:
            trade_dollars = abs(delta) * capital
            cost = trade_dollars * (cfg.FEE_RATE + cfg.SLIPPAGE)
            capital -= cost

            if position == 0.0 and target != 0.0:
                entry_price    = close
                entry_date     = date
                entry_capital  = capital
                entry_features = ml_row

            elif target == 0.0 and position != 0.0 and entry_date is not None:
                exit_pnl = capital - entry_capital if entry_capital else float("nan")
                won      = exit_pnl > 0
                classifier.add_trade(entry_features if entry_features is not None
                                     else pd.Series(), label=won)
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
                    "confidence":  float(classifier.predict(entry_features)
                                         if entry_features is not None else 0.5),
                    "ml_active":   classifier.is_ready(),
                })
                entry_price = entry_date = entry_capital = entry_features = None

            position = target

        equity_values.append({"date": date, "capital": capital})

    equity_curve = pd.DataFrame(equity_values).set_index("date")["capital"]
    trades = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
        columns=["entry_date", "exit_date", "direction",
                 "entry_price", "exit_price", "entry_z", "exit_z",
                 "pnl", "pnl_pct", "confidence", "ml_active"]
    )
    confidence_log = pd.DataFrame(confidence_records).set_index("date") if confidence_records else pd.DataFrame()

    # Open / unrealized trade (mirrors engine.run_backtest behaviour)
    import numpy as np
    open_trade: dict | None = None
    if entry_date is not None and entry_price is not None and position != 0.0:
        last_date  = df.index[-1]
        last_close = float(df["close"].iloc[-1])
        last_z     = float(df["zscore"].iloc[-1]) if not np.isnan(df["zscore"].iloc[-1]) else float("nan")
        unrealized_pct = (last_close / entry_price - 1.0) if entry_price else float("nan")
        open_trade = {
            "entry_date":     entry_date,
            "last_date":      last_date,
            "direction":      "LONG" if position > 0 else "SHORT",
            "entry_price":    entry_price,
            "last_price":     last_close,
            "entry_z":        float(df.at[entry_date, "zscore"]) if entry_date in df.index else float("nan"),
            "current_z":      last_z,
            "position_size":  position,
            "unrealized_pnl": unrealized_pct * abs(position) * (entry_capital or capital),
            "unrealized_pct": unrealized_pct,
        }

    print(f"  ML classifier: {classifier.n_trades} trades seen, "
          f"{'active' if classifier.is_ready() else 'not yet active'}")
    if classifier.is_ready() and classifier.feature_importances:
        top = sorted(classifier.feature_importances.items(), key=lambda x: -x[1])
        print(f"  Top features: " + "  ".join(f"{k}={v:.3f}" for k, v in top[:4]))

    return equity_curve, trades, open_trade, confidence_log
