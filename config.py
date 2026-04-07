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

# Macro shock filter: scale position DOWN as the asset draws down from its recent
# high, rather than applying a hard binary on/off gate.
# Position multiplier: 1.0 at 0% drawdown, linear to 0.0 at MACRO_DD_HARD.
# The original binary threshold at -40% would have nearly blocked the March 2020
# BTC entry ($6,199 = ~-41% from the 90-day high) — a hard gate at exactly the
# best entry point.  A continuous scale-down is more robust.
USE_MACRO_FILTER     = True
MACRO_DD_SOFT        = -0.20   # drawdown at which scaling begins (full position)
MACRO_DD_HARD        = -0.50   # drawdown at which position reaches zero
MACRO_DD_WINDOW      = 90      # days for the rolling high

# ── Position sizing ────────────────────────────────────────────────────────────

LONG_ONLY     = True        # False → allow short positions
MAX_POSITION  = 1.00        # maximum fraction of capital deployed (1.0 = fully invested)
ZSCORE_SCALE  = 3.0         # |Z| at which position is at MAX_POSITION

# Vol-targeting overlay: scale position to maintain constant portfolio volatility.
# At each rebalance, the position multiplier = min(1, VOL_TARGET / realised_vol).
# Example: VOL_TARGET=0.40, realised_vol=0.80 → position capped at 50% of MAX_POSITION.
#          VOL_TARGET=0.40, realised_vol=0.15 → multiplier = 1.0 (no reduction).
# BTC realised daily vol at 120% annualised → position caps at ~33%.
# This prevents ruin-level dollar exposure during volatility spikes without
# requiring a hard drawdown stop.
USE_VOL_TARGET     = True   # enable vol-targeting multiplier
VOL_TARGET_ANNUAL  = 0.40   # target annualised portfolio volatility (40%)
VOL_LOOKBACK_DAYS  = 30     # trailing window for realised vol estimate

# Minimum position change to trigger a rebalance (avoids churning on tiny moves).
REBALANCE_BAND = 0.05

# ── Backtest ───────────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 10_000.0
FEE_RATE        = 0.001     # 0.10% per trade (one-way)
SLIPPAGE        = 0.0005    # 0.05% per trade (one-way)
START_DATE      = "2016-01-01"

# ── Drawdown circuit breaker ───────────────────────────────────────────────────
# If the portfolio drops more than MAX_PORTFOLIO_DD from its all-time peak,
# force the position to zero.  The circuit resets once the equity recovers
# CIRCUIT_BREAKER_RESET above the trough (prevents re-entering into a continuing
# bear).  BTC has had three >70% drawdowns since 2013; without this, a 100%
# allocation into a prolonged bear is ruin-level risk.
MAX_PORTFOLIO_DD       = -0.60    # -60% from equity peak triggers the breaker
CIRCUIT_BREAKER_RESET  =  0.10   # recover 10% from trough before re-enabling

# ── Execution realism ──────────────────────────────────────────────────────────

# Execute at next bar's open (T+1) instead of same-bar close.
# Eliminates look-ahead on execution price; reduces CAGR slightly but is
# more realistic for EOD strategies. Disable with --no-t1 for comparison.
T_PLUS_ONE = True

# ── Tax modeling ──────────────────────────────────────────────────────────────
# Rates are composited as: effective = federal + state + niit (capped at 100%).
# Federal rates: short-term 37% (top bracket), long-term 20% (top bracket).
# NIIT (Net Investment Income Tax): 3.8% on investment income above thresholds
#   ($200k single / $250k married) — applies to crypto gains.
# State: 0% default; set to your state rate (CA = 13.3%, NY = 10.9%, TX = 0%).
# At the marginal rates for a high-income trader running this strategy, the
# effective short-term rate in California is ~54% (37 + 13.3 + 3.8).
APPLY_TAXES      = False   # off by default; enable with --taxes flag
TAX_SHORT_TERM   = 0.37    # federal rate for gains held < 1 year (top bracket)
TAX_LONG_TERM    = 0.20    # federal rate for gains held >= 1 year (top bracket)
TAX_NIIT_RATE    = 0.038   # Net Investment Income Tax (3.8%); set 0 if below threshold
TAX_STATE_RATE   = 0.00    # state rate; e.g. 0.133 for California, 0.109 for NY

# ── Demand index ───────────────────────────────────────────────────────────────

# Enable demand-layer filters in the strategy
USE_DEMAND_FILTER = False   # gate entries: only enter when demand is rising
USE_DEMAND_EXIT   = False   # enhance exits: exit early when demand rolls over

# Component weights for the composite demand index.
# This is the single source of truth — imported by models/demand_index.py.
# Weights auto-renormalize when a component is unavailable (e.g. no API key).
# Set a weight to 0 to permanently exclude a source.
DEMAND_WEIGHTS: dict[str, float] = {
    "trends":       0.20,   # Google Trends composite (attention)
    "volume":       0.15,   # Spot dollar volume vs rolling avg (action)
    "mvrv":         0.15,   # MVRV Z-score inverted (intent) — requires on-chain data
    "etf":          0.15,   # ETF dollar volume (institutional action)
    "outflows":     0.05,   # Exchange outflows (accumulation intent) — requires on-chain data
    "fear_greed":   0.15,   # Fear & Greed Index momentum (sentiment)
    "active_addr":  0.10,   # Active addresses (adoption/network usage)
    "exchange_vol": 0.05,   # Multi-exchange aggregated volume (ccxt)
}

# Normalization window for each demand component (rolling Z-score days)
DEMAND_NORM_WINDOW = 90

# Google Trends queries
TRENDS_QUERIES = ["bitcoin", "buy bitcoin", "crypto"]
TRENDS_GEO     = ""   # "" = worldwide; "US" = United States only
