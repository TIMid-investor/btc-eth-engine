"""
config.py — All tunable parameters for the crypto swing trading engine.

Edit these values to calibrate the model. Every other module imports from here
so there is a single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Symbols ────────────────────────────────────────────────────────────────────

BTC_SYMBOL = "BTC-USD"
ETH_SYMBOL = "ETH-USD"

# Reference dates used as t=0 for the power-law time axis.
# Using genesis block / mainnet launch so the exponent is consistent over
# the full history of each asset.
BTC_GENESIS = "2009-01-03"
ETH_GENESIS = "2015-07-30"

# ── Data cache ─────────────────────────────────────────────────────────────────

DATA_CACHE = Path(os.environ.get("CRYPTO_DATA_DIR", str(Path.home() / "crypto-data")))

# ── Curve model ────────────────────────────────────────────────────────────────

# "power_law" → fit price = a * days_since_genesis ^ b in log-log space
# "log_ema"   → long-term EMA of log(price) as the expected curve
CURVE_MODEL = "power_law"

LOG_EMA_SPAN = 730          # days; used when CURVE_MODEL = "log_ema"

# ── Z-score ────────────────────────────────────────────────────────────────────

# Rolling window (days) for the residual standard deviation.
# Shorter = more adaptive; longer = more stable.
ZSCORE_WINDOW      = 365
ZSCORE_MIN_PERIODS = 180    # minimum history before the signal is trusted

# ── Signal thresholds ─────────────────────────────────────────────────────────

BUY_THRESHOLD  = 1.5        # |Z| must exceed this on the downside  → go long
SELL_THRESHOLD = 1.5        # |Z| must exceed this on the upside   → go flat / short
EXIT_THRESHOLD = 0.25       # close the position when |Z| falls inside this band

# ── Hold-through-cycle mode ────────────────────────────────────────────────────
# When True the backtest ignores continuous Z-score position sizing and instead
# runs a stateful buy/hold/sell machine:
#   ENTER  (full MAX_POSITION) when Z drops below -BUY_THRESHOLD  (buy the dip)
#   HOLD                       while Z is between -BUY_THRESHOLD and SELL_THRESHOLD
#   EXIT                       when Z rises above  SELL_THRESHOLD (overbought)
# This captures full bull-market runs (e.g. Aug 2020 → Apr 2021) rather than
# selling out as Z mean-reverts to zero.  All filters still gate the entry.
HOLD_THROUGH_CYCLE = False

# ── Filters ────────────────────────────────────────────────────────────────────

USE_TREND_FILTER  = True
TREND_EMA_DAYS    = 140     # 20-week EMA slope; long only when slope is positive

USE_VOLUME_FILTER = True
VOLUME_WINDOW     = 30      # days for the volume rolling average
# A buy signal requires volume to be above this fraction of the 30d avg.
# High deviation + low volume = weaker conviction.
VOLUME_MIN_RATIO  = 0.80

# Macro shock filter: suppress signals when the asset is in a severe drawdown
# (assumes the move is macro-driven, not mean-reverting around the curve).
USE_MACRO_FILTER     = True
MACRO_DD_THRESHOLD   = -0.40   # drawdown from rolling 90-day high
MACRO_DD_WINDOW      = 90      # days for the rolling high

# ── Position sizing ────────────────────────────────────────────────────────────

LONG_ONLY     = True        # False → allow short positions
MAX_POSITION  = 1.00        # maximum fraction of capital deployed (1.0 = fully invested)
ZSCORE_SCALE  = 3.0         # |Z| at which position is at MAX_POSITION

# Minimum position change to trigger a rebalance (avoids churning on tiny moves).
REBALANCE_BAND = 0.05

# ── Backtest ───────────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 10_000.0
FEE_RATE        = 0.001     # 0.10% per trade (one-way)
SLIPPAGE        = 0.0005    # 0.05% per trade (one-way)
START_DATE      = "2016-01-01"
