# Crypto Engine — Changelog

All changes are logged here with date, what changed, and why.
Findings that motivated changes are in `crypto_critic.md` (critique) and the
research report produced on 2026-04-03.

---

## 2026-04-03

### [1] Always show buy-and-hold benchmark in backtest output
**File:** `scripts/run_backtest.py`
**Why:** The strategy's 11.1% CAGR reads as impressive in isolation. Alongside
the ~50% buy-and-hold CAGR it is correctly framed as a risk-reduction tool, not
an alpha generator. Every backtest output now shows B&H CAGR, B&H max drawdown,
and strategy excess return vs B&H side-by-side.

---

### [2] Report open/unrealized trade in backtest output
**File:** `backtest/engine.py`, `scripts/run_backtest.py`
**Why:** The trades table only showed closed trades, making the win rate of 100%
in the hold-through-cycle mode misleading when a large open position (e.g., the
2022 entry) was unresolved. The engine now returns an `open_trade` dict alongside
`trades`, and run_backtest.py surfaces it with entry date, entry Z, current Z,
and unrealized P&L.

---

### [3] Add drawdown circuit breaker
**File:** `config.py`, `backtest/engine.py`
**Why:** MAX_POSITION=1.0 with no drawdown stop means 100% exposure into a 70%+
BTC bear market — ruin-level risk. BTC has had three drawdowns exceeding 70%
since 2013. Added MAX_PORTFOLIO_DD (default -0.60) and CIRCUIT_BREAKER_RESET
(default 0.10). When portfolio drops >60% from peak, position is forced to zero
until a fresh signal fires after a 10% recovery from the trough.

---

### [4] Replace binary macro filter with continuous scale-down
**File:** `config.py`, `models/filters.py`, `backtest/engine.py`
**Why:** The binary -40% threshold fires as a cliff-edge gate. Research showed
the filter likely blocked the March 2020 entry (BTC was ~-41% from its 90-day
high at $6,199 on March 20, 2020). A continuous multiplier starting at
MACRO_DD_SOFT (default -0.20) and reaching zero at MACRO_DD_HARD (default -0.50)
scales position size proportionally rather than blocking it entirely. Replaced
`macro_ok` boolean with `macro_mult` float [0, 1] applied as a position multiplier.

---

### [5] Add NIIT and state tax rate config options
**File:** `config.py`, `backtest/tax.py`, `scripts/run_backtest.py`
**Why:** TAX_SHORT_TERM=0.37 and TAX_LONG_TERM=0.20 are federal-only placeholders.
Combined federal + state + NIIT rates can exceed 54% in high-tax states. Added
TAX_NIIT_RATE (default 0.038) and TAX_STATE_RATE (default 0.0, user-configurable).
Effective combined rate is now displayed in output when --taxes is enabled.

---

### [6] Add multi-curve robustness test (--curve-compare flag)
**File:** `models/power_law.py`, `scripts/run_backtest.py`
**Why:** R²=0.92 on a log-log regression of two monotonically increasing series
is not evidence the power law is correct — any monotonic function would fit.
Added rolling_median_deviation() to power_law.py and a --curve-compare flag to
run_backtest.py that runs power_law, log_ema, and rolling_median in parallel and
prints results side-by-side. If results collapse on non-power-law curves, the
backtest is a curve-fitting story.

---

### [7] Add walk-forward threshold selection (--walk-forward-params flag)
**File:** `backtest/engine.py`, `scripts/run_backtest.py`
**Why:** BUY_THRESHOLD=1.5 and SELL_THRESHOLD=1.5 were fixed constants presumably
chosen by looking at the full history. The existing parameter sensitivity test
varied thresholds on in-sample data, which is not robustness testing. Added
--walk-forward-params flag: at each refit date, the engine selects thresholds
that maximised Sharpe on the trailing 2 years of data, then applies them
forward-only. If optimal threshold clusters at 1.4-1.6 across all windows,
the default is validated; if it varies widely, the threshold is noise.

---

### [8] Add IC validation for demand components
**File:** `scripts/run_demand_ic.py` (new)
**Why:** The demand index combines components of unknown predictive value. Google
Trends, Fear & Greed, and ETF volume are likely coincident or lagging signals
(supported by Urquhart 2018 research). Added run_demand_ic.py which computes
Spearman IC between each demand component and 30/60-day forward BTC return on
rolling out-of-sample windows. Components with |IC| < 0.03 are flagged as noise.
This is a prerequisite before enabling USE_DEMAND_FILTER.

---

### [9] Replace ETF flow proxy with shares-outstanding delta
**File:** `data/etf_flows_fetcher.py`
**Why:** The prior flow proxy (dollar_volume × sign(price_change)) was a momentum
signal, not a flow estimate. BlackRock publishes daily shares outstanding for IBIT
(and Fidelity for FBTC) on their fund pages. Daily Δshares × NAV gives a true net
creation/redemption flow. Updated fetcher to compute this as the primary flow
metric; the signed-volume proxy is retained as a fallback when shares outstanding
data is unavailable.

---

### [10] Add vol-targeting position overlay
**File:** `config.py`, `backtest/engine.py`
**Why:** 100% allocation during a BTC volatility spike (120% annualised vol) means
massive dollar risk per unit of time. Standard vol-targeting scales position to
maintain constant portfolio volatility. Added VOL_TARGET_ANNUAL (default 0.40),
VOL_LOOKBACK_DAYS (default 30), and USE_VOL_TARGET (default True). The multiplier
= min(1, VOL_TARGET / realised_vol). At 15% vol → 100% position. At 120% vol →
33% position. Applied after all other filters in _compute_target_position().

---

### [11] Add OU forecast parameter uncertainty bands
**File:** `scripts/run_forecast.py`
**Why:** The OU forecast fan chart showed path uncertainty only, implying a
precision that doesn't exist. The 95% CI on the mean-reversion speed θ from a
single 3000-day series is approximately ±50%. Added a second shaded band
representing the θ±50% parameter uncertainty range, labelled "param uncertainty"
in the legend. Viewers can now distinguish model uncertainty (inner band) from
parameter uncertainty (outer band).

---
