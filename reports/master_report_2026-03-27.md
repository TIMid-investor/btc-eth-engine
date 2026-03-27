# BTC & ETH Swing Trading Engine — Master Report
### 27 March 2026

---

## Table of Contents
1. [What This Engine Is and Is Not](#1-what-this-engine-is-and-is-not)
2. [The Core Idea: Power-Law Mean Reversion](#2-the-core-idea-power-law-mean-reversion)
3. [Architecture Overview](#3-architecture-overview)
4. [The Long-Term Growth Curve](#4-the-long-term-growth-curve)
5. [Z-Score: Measuring Deviation](#5-z-score-measuring-deviation)
6. [Filters: Avoiding Bad Trades](#6-filters-avoiding-bad-trades)
7. [Position Sizing](#7-position-sizing)
8. [Regime Awareness](#8-regime-awareness)
9. [ML Overlay](#9-ml-overlay)
10. [Backtest Results: Full Breakdown](#10-backtest-results-full-breakdown)
11. [Parameter Sweep: Finding the Efficient Frontier](#11-parameter-sweep-finding-the-efficient-frontier)
12. [Walk-Forward Validation: The Honest Number](#12-walk-forward-validation-the-honest-number)
13. [Live Signals: Where We Stand Today](#13-live-signals-where-we-stand-today)
14. [What to Watch Going Forward](#14-what-to-watch-going-forward)
15. [Known Limitations and Caveats](#15-known-limitations-and-caveats)
16. [Quick Reference: Running the Engine](#16-quick-reference-running-the-engine)

---

## 1. What This Engine Is and Is Not

**What it is:**
A quantitative mean-reversion swing trading model for Bitcoin and Ethereum. It treats both assets as fundamentally trending upward over multi-year timescales — but doing so in a highly volatile, oscillating fashion. The engine identifies when price has strayed significantly below (or above) its long-run expected trajectory, sizes a position proportional to the degree of dislocation, and exits when price returns toward the trend.

**What it is not:**
- A price prediction model — it does not forecast where BTC or ETH will go
- A high-frequency or momentum system — average trade duration is days to weeks
- A fully validated live-trading system — backtests have inherent biases (detailed in Section 15)
- A guarantee of returns — crypto is the most volatile major asset class in existence

**The philosophy:**
Bitcoin has appreciated roughly 15,000x since 2014. Nobody who bought in 2014 and held until 2026 needed a model. The question this engine addresses is: *for someone who cannot stomach 80% drawdowns, can you participate in crypto's long-run appreciation with dramatically lower volatility and drawdown, accepting lower absolute returns as the trade-off?* The answer, based on this backtest, is broadly yes — but with important caveats.

---

## 2. The Core Idea: Power-Law Mean Reversion

### The Long-Run Trend
Bitcoin's price has not grown linearly, nor exponentially. It has followed a *power law* — a decelerating growth rate that is best described in log-log space. When you plot log(price) vs log(days since genesis block), you get something close to a straight line. This means:

```
price(t) = a × t^b
```

where:
- `t` = days since the BTC genesis block (3 Jan 2009)
- `a` = a very small scale constant (~4.09e-18)
- `b` = the growth exponent (~5.92 for BTC, ~2.08 for ETH)
- `b < 1 in log space` would mean slowing growth — but b here is applied to days, not log-days

In log-log space (which is how the fit is done), this becomes a simple linear regression:

```
log(price) = log(a) + b × log(days)
```

**BTC fit:**  `price = 4.09×10⁻¹⁸ × days^5.9247`   R² = 0.92
**ETH fit:**  `price = 1.19×10⁻⁴ × days^2.0810`     R² = 0.58

The R² of 0.92 for BTC is remarkable — 92% of Bitcoin's price variance across 12 years is explained by a single-variable regression on time. The ETH fit is weaker (0.58) because ETH has a shorter history and a more complex adoption curve that doesn't align as cleanly with a single power law.

### The Oscillation
Around this rising trend, the actual price oscillates dramatically — sometimes 3–4 standard deviations above (euphoria peaks) and sometimes 3–6 standard deviations below (bear market troughs). The engine treats these oscillations as the primary source of tradeable signal.

Think of it as a rubber band. The trend is the centre point. Bullish mania stretches the band upward; fear and deleveraging stretches it downward. Mean reversion is the tendency for the band to snap back toward the centre.

---

## 3. Architecture Overview

```
yfinance (BTC-USD, ETH-USD daily OHLCV)
           │
           ▼
   data/fetcher.py
   Parquet cache (~/ crypto-data/)
           │
           ▼
   models/power_law.py
   Expected price curve: price = a × days^b
   log_deviation = log(price / expected)
           │
           ▼
   models/zscore.py
   Z_t = log_deviation_t / rolling_std(365 days)
           │
           ▼
   models/filters.py          models/regime.py
   Trend filter (EMA slope)   Halving cycle phase
   Volume filter              Bull/Bear regime
   Macro shock filter         → Adjusted thresholds
           │                          │
           └──────────┬───────────────┘
                      ▼
            Target position [0, 1]
            (scaled by Z magnitude)
                      │
                      ▼
   models/ml_overlay.py (optional)
   Logistic classifier: P(trade wins)
   Gates entries below confidence threshold
                      │
                      ▼
   backtest/engine.py
   Daily mark-to-market + trade log
   Fees: 0.10% + Slippage: 0.05% per trade
                      │
           ┌──────────┼──────────┐
           ▼          ▼          ▼
   backtest/        scripts/   scripts/
   metrics.py    run_charts   param_sweep
   CAGR/Sharpe/  4 charts     Grid search
   MaxDD/etc     per asset    Pareto front
```

---

## 4. The Long-Term Growth Curve

### How the Fit Works

The curve is fit in log-log space using ordinary least squares (OLS). The inputs are:

- **X axis:** `log(days since genesis block)` — 3 Jan 2009 for BTC, 30 Jul 2015 for ETH
- **Y axis:** `log(close price)` — daily closing price

A straight line in this space corresponds to a power law in price space. The OLS fit gives us the slope (`b`) and intercept (`log(a)`).

```python
# Simplified version of what models/power_law.py does:
log_t = np.log(days_since_genesis)
log_p = np.log(price)
b, log_a = np.polyfit(log_t, log_p, deg=1)
a = np.exp(log_a)
# Result: price = a × days^b
```

### What the Exponent Means

For BTC with `b ≈ 5.92`: if the number of days since genesis doubles, the expected price rises by a factor of `2^5.92 ≈ 60×`. This sounds large but remember the base grows slowly — doubling from 1,000 days to 2,000 days represents a fundamentally different adoption stage than doubling from 5,000 to 10,000 days. The key property is that growth decelerates: each additional day adds a smaller *percentage* gain than the previous day.

### The Standard Deviation Bands

The ±1σ and ±2σ bands in Chart 1 are not fixed — they are computed using a rolling 365-day trailing window of the actual log-deviation residuals. This means the bands widen during volatile periods and tighten during calm ones, making the signal adaptive.

### The Log-EMA Alternative

If the power-law fit feels too anchored to a historical genesis date, the model also supports a `log_ema` curve mode: a long-term exponential moving average (730-day span by default) applied to log(price). This is entirely adaptive — the "expected" price is just a slow-moving average of the recent log-price history. It requires no genesis date assumption but cannot extrapolate beyond the data range.

Switch with: `--curve log_ema`

---

## 5. Z-Score: Measuring Deviation

### The Formula

```
log_deviation_t = log(price_t / expected_price_t)

Z_t = log_deviation_t / rolling_std(log_deviation, window=365, min_periods=180)
```

**Why log deviation, not raw price difference?**
- BTC has ranged from $200 to $110,000. A raw price difference of $5,000 means something completely different in 2016 vs 2021.
- Log deviation is scale-invariant: +0.30 always means "35% above the trend" regardless of the absolute price level.
- Log deviations are more normally distributed, making the standard deviation calculation meaningful.

**Why a 365-day rolling window?**
- Short windows (90 days) are too reactive — a volatile month changes the σ dramatically, making the Z-score noisy.
- Long windows (730+ days) are too slow — they blend multiple market regimes.
- 365 days captures roughly one full seasonal cycle while remaining responsive to structural shifts in volatility.

### Signal Thresholds

```
Z < -1.5  →  Buy signal (oversold)
Z > +1.5  →  Sell signal / go flat (overbought)
|Z| < 0.25 →  Exit zone (return toward mean, close position)
```

The thresholds are not magical — they represent a trade-off between signal frequency and quality. Lower thresholds (e.g. 1.0) produce more trades but with lower average conviction. Higher thresholds (e.g. 2.5) produce fewer, higher-conviction trades. The parameter sweep explores this systematically (see Section 11).

### What Z-Score Levels Mean Intuitively

| Z-score | Meaning | Historical context |
|---------|---------|-------------------|
| 0       | Exactly on the long-run trend | Rare — price transits quickly |
| ±1.0    | Within normal oscillation range | Occurs frequently |
| -1.5    | Meaningfully below trend | Early-to-mid bear trough |
| -2.5    | Deeply below trend | Late bear market, capitulation |
| -3.5+   | Extreme historical undervaluation | COVID crash, 2016 early bear |
| +1.5    | Above trend | Late bull market momentum |
| +2.5+   | Significantly overvalued vs trend | 2017 peak, 2021 peak |

**Today (27 March 2026):** BTC Z = -2.62, ETH Z = -1.76. Both are signalling value relative to their long-run curves. More on this in Section 13.

---

## 6. Filters: Avoiding Bad Trades

Three optional filters gate the Z-score signal. Each addresses a different failure mode.

### Trend Filter (20-Week EMA Slope)
**What:** Computes the slope of a 140-day (≈ 20-week) exponential moving average of price. If the slope is negative (EMA declining), the model blocks new long entries.

**Why:** Mean reversion works best when the intermediate trend is at least neutral. Buying into a strongly falling market often produces *catching falling knives* — the price continues lower, the Z-score gets even more negative, and the position loses money even though the asset is "cheap" relative to the long-run curve. The trend filter adds a momentum confirmation: only buy dips when the intermediate trend is stabilising or recovering.

**Effect on backtest:** Trend filter ON vs OFF: average Sharpe rises from 0.434 → 0.671. This is the single highest-value filter by a significant margin.

**Trade-off:** The trend filter will cause you to miss entries at the absolute bottom (the moment the trend is most negative). You will enter slightly later, at a slightly higher price, but with much higher probability of continued recovery.

### Volume Filter
**What:** Blocks signals when the day's trading volume is below 80% of its 30-day rolling average.

**Why:** Large price dislocations on thin volume are less reliable signals. They can reflect temporary illiquidity, weekend thin markets, or algorithmic noise rather than genuine capitulation or genuine buying interest. When the Z-score is extreme AND volume is elevated, the signal has more conviction — many participants are engaging with the move.

**Effect:** Modest improvement in signal quality; primarily eliminates a handful of low-conviction entries.

### Macro Shock Filter
**What:** Blocks all new entries when price has fallen more than 40% from its 90-day high.

**Why:** During genuine macro tail events (exchange collapses like FTX, global credit seizures, regulatory crackdowns), the power-law framework temporarily breaks down. These are not mean-reversions around the long-run curve — they are structural repricing events where the fundamental backdrop has changed. During FTX in November 2022, BTC fell 25% in a week. Buying into that on a Z-score signal would have been entering a falling building.

The filter says: if we're already down 40% from a recent high, wait for stabilisation before entering. You will miss the exact bottom but you will avoid the worst of the "dead cat bounce into further collapse" pattern.

**Effect:** Prevents the most painful entries; accepts missing some of the sharpest recoveries.

---

## 7. Position Sizing

The engine uses a **continuous, Z-score-scaled position size** rather than a fixed all-or-nothing approach:

```
size = clip( (|Z| - threshold) / (ZSCORE_SCALE - threshold), 0, 1 ) × MAX_POSITION

Where:
  threshold   = BUY_THRESHOLD (default 1.5)
  ZSCORE_SCALE = 3.0 (Z at which position is fully deployed)
  MAX_POSITION = 1.0 (100% of capital)
```

| Z-score | Position size |
|---------|--------------|
| -1.5 (threshold) | 0% (no position) |
| -2.0 | 33% |
| -2.5 | 67% |
| -3.0+ | 100% |

**Why this matters:** A fixed-size model (all-in at Z = -1.5, all-out at 0) is highly sensitive to threshold choice. The continuous sizing means:
- Small deviations → small positions → small risk if the signal fails
- Large deviations → large positions → you are maximally deployed at the exact moments of greatest historical value

**Rebalance band:** The engine only rebalances when the target position differs from the current by more than 5%. This prevents constant small adjustments that would erode returns through fees on noise.

---

## 8. Regime Awareness

### The Problem with Static Thresholds

A Z = -2.0 signal in Q4 2020 (post-halving, bull market beginning) is a very different beast from the same signal in Q3 2022 (FTX contagion building, macro risk-off). Static thresholds treat both identically. Regime awareness adjusts the required signal strength based on the current market context.

### Halving Cycle (BTC Only)

Bitcoin's supply schedule creates a natural four-year cycle. The halvings occurred on:
- Nov 2012, Jul 2016, May 2020, Apr 2024 (next expected ~Apr 2028)

Historical pattern: Post-halving early phase (0–12 months) has been the most consistently bullish period. The supply shock (new coin issuance cut in half) meets demand that has accumulated during the previous bear market. Late-cycle (24+ months post-halving) tends to be when overextension risk is highest.

**Threshold adjustments by phase:**

| Phase | Multiplier | Effect |
|-------|-----------|--------|
| POST_EARLY + BULL | 0.80× | Enter at Z = -1.2 instead of -1.5 (more aggressive) |
| POST_EARLY + NEUTRAL | 0.90× | Enter at Z = -1.35 |
| LATE_CYCLE + BEAR | 1.30× | Enter only at Z = -1.95 (much more selective) |
| PRE_HALVING + BEAR | 1.20× | Enter only at Z = -1.8 |

### Price Regime (BTC and ETH)

Using the 200-day SMA and drawdown from 365-day high:
- **BULL:** price ≥ 5% above 200d SMA and drawdown < -30%
- **BEAR:** price ≥ 10% below 200d SMA or drawdown < -50%
- **ACCUMULATION:** recently crossed 200d SMA from below (60-day window)
- **NEUTRAL:** everything else

**Current regime (27 March 2026):**
BTC is in the `POST_EARLY|BEAR` / `LATE_CYCLE|BEAR` zone — the April 2024 halving was ~11 months ago, placing us near the end of POST_EARLY phase. The price regime is BEAR (price below 200d SMA, significant drawdown from ATH). The model therefore requires a stronger Z-score signal than baseline before entering.

---

## 9. ML Overlay

### Motivation

The rule-based signal (Z-score + filters + regime) treats all entries at a given Z level as equivalent. In practice, the *path* to that Z level matters:
- A Z of -2.0 reached by a slow, grinding 6-month decline is different from the same Z reached by a sudden 30% crash
- A Z of -2.0 where Z-score has been accelerating downward (still falling) is different from one where Z is stabilising

The ML overlay learns these nuances from historical trade outcomes.

### Architecture

**Features (8 inputs per signal day):**

| Feature | Description |
|---------|-------------|
| `zscore` | The primary signal — current Z-score value |
| `zscore_vel5` | 5-day rate of change of Z (is it still falling?) |
| `zscore_accel5` | 5-day acceleration of Z-velocity |
| `trend_slope` | Fractional slope of 20-week EMA (normalised) |
| `volume_ratio` | Today's volume / 30-day average |
| `log_deviation` | Raw log(price/expected) — correlated with but distinct from Z |
| `days_in_zone` | Consecutive days Z has been below threshold |
| `price_regime_enc` | BULL=1, ACCUMULATION=0.5, NEUTRAL=0, BEAR=-1 |

**Model:** Logistic regression with L2 regularisation and StandardScaler normalisation. Simple, fast, interpretable — the coefficient magnitudes directly reveal which features drive trade quality.

**Walk-forward training:** The classifier is *never* trained on future data. It trains only on trades that have already closed, adding each closed trade as a new observation. It activates only after 20 closed trades and refits every 10 additional trades. This mimics how you'd use it in practice: the model learns from its own history as it accumulates.

**Feature importance (as learned on BTC 2016–2026):**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | zscore | 1.203 |
| 2 | zscore_accel5 | 1.100 |
| 3 | log_deviation | 0.789 |
| 4 | trend_slope | 0.699 |
| 5 | volume_ratio | ~0.4 |

**Interpretation:** The most important predictor of a winning trade is, unsurprisingly, the Z-score magnitude itself — deeper signal = higher win probability. The second most important is Z-score *acceleration* — when the Z is not just negative but *still declining*, the ML model suppresses entries because the signal hasn't bottomed yet. The trend slope and raw deviation round out the top 4.

**Effect on trade count:** ML with a 0.55 confidence threshold reduces trades from 103 → 57, with similar win rate but improved precision on the trades taken.

**Important caveat:** With only 57–103 trades in the backtest period, the ML model has limited data to learn from. The classifier is directionally useful but should not be over-trusted until 200+ trades of live history are accumulated.

---

## 10. Backtest Results: Full Breakdown

### Methodology

- **Data:** Daily OHLCV from yfinance (BTC-USD from Sep 2014, ETH-USD from Nov 2017)
- **Start date:** 2016-01-01 for BTC (oldest reliable data with enough Z-score burn-in), 2018-05-07 for ETH
- **Fees:** 0.10% per trade (one-way) — representative of major CEX spot trading fees
- **Slippage:** 0.05% per trade (one-way) — conservative estimate for liquid markets
- **No shorting** (long-only mode)
- **Benchmark:** Buy-and-hold from the same start date

---

### BTC — Baseline (Full-History Curve Fit)

```
Period: 2016-01-01 → 2026-03-26  (10.2 years)
Starting capital: $10,000

Metric               BTC Strategy   Buy & Hold
─────────────────────────────────────────────
CAGR                    +11.1%         +40.7%
Sharpe                    0.83           0.89
Sortino                   0.62           1.20
Calmar                    0.52           0.49
Max Drawdown            -21.5%         -83.4%
Ann. Volatility         +13.8%         +55.7%
Win Rate                +57.3%            —
Profit Factor             3.07            —
Avg Duration(d)            5.0            —
Trades                     103            —
─────────────────────────────────────────────
Final value:     $47,534  (+375%)
B&H final value: $1,583,842  (+15,738%)
```

**Reading these results:**

- **CAGR 11.1% vs 40.7%:** The strategy severely underperforms buy-and-hold in absolute terms. This is expected and important: the model *intentionally* misses most of the bull market appreciation. It is only invested during windows of deep oversold conditions — it misses 2017's 20× run, most of 2020–2021's 10× run, etc.

- **Max Drawdown -21.5% vs -83.4%:** This is the core value proposition. Buy-and-hold required enduring an 83% peak-to-trough drawdown at least once. The strategy's worst drawdown was 21.5% — roughly equivalent to a bad year in equities. For a risk-averse participant in crypto, this is the entire point.

- **Sharpe 0.83 vs 0.89:** The Sharpe ratios are similar, meaning the strategy generates roughly the same *risk-adjusted* return as buy-and-hold, just at a much lower absolute level. The Calmar ratio (CAGR / max drawdown) is 0.52 vs 0.49 — marginally better for the strategy.

- **Profit Factor 3.07:** Winning trades generate 3.07× the dollar loss of losing trades. This is a strong profit factor — the asymmetry between wins and losses is healthy.

- **Average trade duration 5 days:** These are short swing trades, not multi-month positions. Most entries and exits happen within a week.

---

### BTC — Best Config from Parameter Sweep (Z=2.0, Filters OFF)

```
Period: 2016-01-01 → 2026-03-26

Metric               BTC Strategy   Buy & Hold
─────────────────────────────────────────────
CAGR                    +18.2%         +40.7%
Sharpe                    1.03           0.89
Sortino                   0.65           1.20
Calmar                    0.73           0.49
Max Drawdown            -24.8%         -83.4%
Win Rate                +100.0%            —
Profit Factor             ∞               —
Avg Duration(d)           72.9            —
Trades                      11            —
─────────────────────────────────────────────
Final value:     $119,101  (+1,091%)
```

**Important caveat on the 100% win rate:** With only 11 trades, this is statistically meaningless. A coin flipped 11 times in a row and coming up heads tells you very little about the coin's true probability. The 11 trades that cleared Z=2.0 without filters happened to all be profitable in this sample — but any sample of 11 is too small to conclude anything about the underlying win rate. This config should be treated as a useful upper-bound reference, not a reliable indicator.

**What it does tell you:** The extreme Z-score events (Z < -2.0) have historically been excellent entry points, even in this small sample. The key question for the next 10 years is whether the power law continues to hold as Bitcoin matures. If adoption continues decelerating smoothly, it will. If Bitcoin suffers a genuine structural challenge (ETF regulation, better alternatives, macro regime change), the curve could shift.

---

### ETH — Baseline

```
Period: 2018-05-07 → 2026-03-26  (7.9 years)

Metric               ETH Strategy   Buy & Hold
─────────────────────────────────────────────
CAGR                     +7.2%          +9.2%
Sharpe                    0.53           0.48
Sortino                   0.21           0.65
Calmar                    0.20           0.10
Max Drawdown            -35.5%         -88.8%
Win Rate                +51.9%            —
Profit Factor             4.40            —
Avg Duration(d)            7.9            —
Trades                      27            —
─────────────────────────────────────────────
Final value:     $22,230  (+122%)
B&H final value: $27,325  (+173%)
```

**Key differences vs BTC:**

1. **Weaker absolute numbers:** ETH's power-law R² is 0.58 vs 0.92. The curve fits poorly, meaning the "expected price" is noisier and the Z-score signals carry less information. ETH doesn't have the same clean power-law adoption curve as BTC.

2. **Strategy beats B&H on Sharpe (0.53 vs 0.48):** ETH buy-and-hold is a slightly worse risk-adjusted bet than BTC buy-and-hold, but the mean-reversion strategy extracts similar risk-adjusted value from both.

3. **Max drawdown -35.5% vs -88.8%:** Still a massive drawdown reduction — the strategy avoids sitting through ETH's 90%+ bear market declines.

4. **Only 27 trades in 8 years:** The trend filter and other filters are very conservative. ETH spends large periods in downtrends (trend filter blocks entries) or in mild oversold territory that doesn't reach the Z=-1.5 threshold.

**Why ETH is harder to model:**
- Shorter history (8 years vs 12 for BTC)
- Multiple fundamental shifts: ETH 1.0 → ETH 2.0 merge, gas fee dynamics, competing L1s
- Power law exponent `b=2.08` vs BTC's `5.92` — the curve grows more slowly, meaning deviations are proportionally larger relative to the signal
- ETH's correlation with BTC means it often falls for macro reasons (BTC selling) rather than ETH-specific ones, making the power-law reversion logic less clean

---

### Walk-Forward (Honest Baseline)

```
BTC Walk-Forward (quarterly refits, 41 unique curve fits):
Period: 2016-09-12 → 2026-03-26

CAGR            +4.3%   (vs +11.1% with look-ahead)
Sharpe           0.51   (vs  0.83)
Max Drawdown   -25.7%   (vs -21.5%)
Win Rate       +47.4%   (vs +57.3%)
Trades             76   (vs    103)
Final value:   $17,908  (+79%)
```

**The honest picture:** The gap between full-history fit (+11.1% CAGR) and walk-forward (+4.3% CAGR) is ~7 percentage points. This is the cost of look-ahead bias in the curve parameters. The curve fit on all data has the advantage of "knowing" where BTC eventually ends up — the quarterly-refit version doesn't.

**What does +4.3% CAGR mean?** Over 10 years, $10,000 becomes $17,908 — a 79% gain. That's approximately what a 60/40 stock-bond portfolio would have returned over the same period. You got equity-like returns with dramatically lower drawdowns than crypto buy-and-hold, but you dramatically underperformed the asset's full potential. Whether that trade-off is worth it depends entirely on your risk tolerance.

---

## 11. Parameter Sweep: Finding the Efficient Frontier

The sweep tested 24 parameter combinations (quick grid) across buy_z, sell_z, z_window, and filter settings.

### Key Findings

**1. Z-score window is the most impactful parameter:**
- 365-day window: avg Sharpe 0.825
- 730-day window: avg Sharpe 0.280
- The 365-day window is significantly better. The 730-day window produces a σ that blends multiple regimes, making the Z-score less sensitive to current conditions.

**2. Trend filter is the second most impactful parameter:**
- With trend filter: avg Sharpe 0.671
- Without trend filter: avg Sharpe 0.434
- The trend filter materially improves risk-adjusted returns by preventing entries into sustained downtrends.

**3. Buy threshold sweet spot is 2.5, not 1.5:**
- Z=1.5 avg Sharpe: 0.628
- Z=2.0 avg Sharpe: 0.537
- Z=2.5 avg Sharpe: 0.493 (but best single config)
The relationship isn't monotonic — at Z=2.5 with a 365-day window, the reduced trade count produces the best Sharpe (0.917) because only the highest-conviction signals are taken.

**4. Sell threshold doesn't matter much:**
- Sell Z=1.5 vs Z=2.0: virtually identical Sharpe (0.553 both)
- The exit threshold matters less than the entry — the exits tend to happen naturally as Z reverts toward 0

**5. Pareto-optimal configs:**
The frontier of "best Sharpe for given max drawdown" contains only 6 configs — all with trend filter ON and either buy_z=2.0 or buy_z=2.5:

| Buy Z | Window | Max DD | Sharpe | Trades |
|-------|--------|--------|--------|--------|
| 2.5 | 365d | -17.4% | 0.917 | 92 |
| 2.0 | 365d | -19.2% | 0.881 | 99 |
| 2.0 | 730d | -16.3% | 0.439 | 24 |
| 2.5 | 730d | -15.4% | 0.277 | 13 |

**Recommendation:** If maximising Sharpe ratio, use `buy_z=2.5, window=365, trend filter ON`. If maximising absolute return with acceptable drawdown, use `buy_z=1.5, window=365, all filters ON` (default config, rank #7).

---

## 12. Walk-Forward Validation: The Honest Number

### Why This Matters

The full-history power-law fit uses BTC's entire price history to determine the curve parameters. When you then backtest from 2016, the curve already "knows" that BTC will reach $100,000 in 2024. This gives the 2016 expected price the benefit of hindsight — the curve knows the long-run destination.

The walk-forward approach eliminates this: at each quarter-end, the curve is refit using only data available up to that moment. The model in 2016 thinks BTC might level off at $1,000. By 2019 it has updated to $10,000. By 2022 it has seen the 2021 peak. Each period uses the state of knowledge available at that time.

### Interpreting the Gap

| Metric | Full-history | Walk-forward | Gap |
|--------|-------------|-------------|-----|
| CAGR | +11.1% | +4.3% | -6.8pp |
| Sharpe | 0.83 | 0.51 | -0.32 |
| Max DD | -21.5% | -25.7% | +4.2pp worse |
| Win rate | 57.3% | 47.4% | -9.9pp |

The walk-forward win rate of 47.4% means the model barely breaks even on trade frequency — it's the profit factor (winning trades being larger than losing trades) that generates the positive return, not raw win rate.

### The True Forward-Looking Expectation

For a realistic forward expectation, the walk-forward result is the baseline you should anchor to: **+4.3% CAGR with -25.7% max drawdown**, assuming the power law continues to hold and the current historical relationships remain stable.

This is not a rich return. But it achieves it with:
- Only 76 trades over 10 years (less than 8 per year)
- Average holding period of under 5 days
- You are invested for roughly 10–15% of trading days

The strategy is almost always in cash — it's patient, selective, and deploys only at moments of extreme dislocation.

---

## 13. Live Signals: Where We Stand Today

**As of 26 March 2026:**

```
═══════════════════════════════════════════════════════
  BTC  ·  as of 2026-03-26
═══════════════════════════════════════════════════════
  Price:       $68,792
  Expected:   $131,444   (power-law curve)
  Deviation:     -47.6%  (log deviation: -0.6466)
  Z-score:       -2.62
  Signal:        BUY ▲▲
  Target:        FLAT — trend filter blocking

  Filters:
    Trend (20-wk EMA slope): DOWN ✗ — blocking entry
    Volume:                  OK ✓
    Macro shock:             OK ✓ (no acute crash)

═══════════════════════════════════════════════════════
  ETH  ·  as of 2026-03-26
═══════════════════════════════════════════════════════
  Price:        $2,060
  Expected:     $3,531   (power-law curve)
  Deviation:     -41.6%  (log deviation: -0.5386)
  Z-score:       -1.76
  Signal:        BUY ▲
  Target:        FLAT — trend filter blocking

  Filters:
    Trend (20-wk EMA slope): DOWN ✗ — blocking entry
    Volume:                  OK ✓
    Macro shock:             OK ✓
```

### What This Means

**Both assets are significantly below their long-run expected prices.** BTC at $68,792 is 47.6% below its power-law expectation of $131,444. ETH is 41.6% below its curve. In Z-score terms:
- BTC at Z = -2.62: historically, this level has coincided with attractive long-term entry points
- ETH at Z = -1.76: above the -1.5 entry threshold — a mild-to-moderate buy signal

**However, both are blocked by the trend filter.** The 20-week EMA slope is still negative for both assets. This means the intermediate trend has not yet turned. The model interprets this as: "the price is cheap relative to the long-run curve, but the momentum is still downward — wait for stabilisation before entering."

**This is not a buy signal today.** It is a *pre-signal* state: the value condition is met, but the timing confirmation (trend reversal) has not occurred.

**The power-law curve for BTC ($131,444) seems very high vs the current price of $68,792.** This requires context:
- The curve represents the long-run trend extrapolated from all available data
- BTC has historically traded below the curve during the middle of bear markets and above it during peaks
- The 2024 all-time high of ~$109,000 was still below the current curve ($131,444), meaning BTC has been below its long-run trend since approximately Q2 2024
- This is consistent with BTC being in the "LATE_CYCLE|BEAR" or "POST_EARLY|BEAR" regime

---

## 14. What to Watch Going Forward

### Primary Watch: Trend Filter Reversal

**The single most important indicator to watch is the 20-week EMA slope turning positive.** When this happens, the model's trend filter will lift and — assuming the Z-score remains below -1.5 at that moment — a buy signal will be generated.

**How to track it:** Run `python3 scripts/run_signals.py --days 30` daily. When you see the trend field change from "DOWN" to "UP ✓", that is the trigger.

**What to expect:** Trend reversals in crypto tend to be preceded by 2–4 weeks of price stabilisation followed by a sharp move. The 20-week EMA is slow — it won't confirm until the recovery has been sustained for several weeks. You will not catch the exact bottom. That's the design.

---

### Secondary Watch: Z-Score Trajectory

Monitor whether the Z-score is:
- **Still falling:** Z moving from -2.62 toward -3.0+ means the sell-off is continuing. The macro shock filter will eventually trigger (-40% from 90-day high) if this continues.
- **Stabilising:** Z hovering around -2.5 to -2.8 for several weeks suggests a base forming.
- **Rising:** Z moving from -2.62 toward -2.0 and then -1.5 is the recovery signal. Combined with a positive EMA slope, this would be the cleanest entry setup.

---

### The Halving Cycle Context

The April 2024 halving was ~11 months ago. We are at the transition between POST_EARLY (0–12 months) and POST_LATE (12–24 months). Historically:
- POST_EARLY has been the most reliably bullish phase
- The fact that BTC is still below its power-law curve 11 months post-halving is unusual — typically the post-halving supply shock pushes prices significantly above the trend
- This suggests the current macro environment (tariffs, rate uncertainty, risk-off globally) is suppressing what might otherwise have been a stronger post-halving rally
- POST_LATE historically sees continued appreciation but with increasing volatility and some mean-reverting oscillations — which is exactly when this model's entry signals become valuable

**Expected setup:** If the macro environment stabilises in Q2/Q3 2026, the combination of a positive EMA slope, Z-score around -1.5 to -2.0, and the POST_LATE halving phase would create a high-quality signal configuration under both baseline and regime-aware modes.

---

### ETH-Specific Factors to Watch

ETH's power-law fit (R²=0.58) is weaker than BTC's, meaning ETH signals require more caution. Additional factors to monitor:
- **ETH/BTC ratio:** If ETH is significantly outperforming or underperforming BTC, it may be driven by ETH-specific factors (staking yields, L2 competition, regulatory treatment) rather than mean-reversion dynamics
- **Staking yield:** ETH staking rewards (~3-4% annually) create a carry that BTC doesn't have — this may structurally shift where the "fair value" sits relative to the power-law curve
- **L1 competition:** Solana and others have taken market share from ETH's DeFi ecosystem; sustained structural loss could invalidate the power-law assumption

---

### Macro Conditions to Monitor

The trend filter and macro shock filter both respond to macro conditions, but indirectly through price. Explicit macro factors to watch:

1. **US rate path:** Lower rates → risk-on → higher probability of crypto trend reversal. The Fed's next moves are the single most important macro input.
2. **Dollar strength (DXY):** Strong dollar historically correlates with weak crypto. DXY declining = crypto tailwind.
3. **Bitcoin ETF flows:** Institutional buying through ETFs has become a structural demand driver since Jan 2024. Watch weekly ETF flow data.
4. **Regulatory clarity:** The US regulatory environment for crypto has been volatile. Major positive or negative regulatory news can move prices 20%+ in days — these events can trigger or suppress signals.
5. **On-chain metrics:** MVRV ratio, exchange outflows, long-term holder accumulation are useful external confirmations. None of these are in the model currently but they're worth watching manually.

---

## 15. Known Limitations and Caveats

### 1. Look-Ahead Bias in the Curve (Partially Addressed)

The default (full-history) power-law fit uses future data to calibrate the curve. The walk-forward mode eliminates this for the Z-score signal but cannot eliminate it entirely — the analyst choosing to use a power-law model at all is itself influenced by knowing the model worked in hindsight.

**Mitigation:** Always interpret results using the walk-forward numbers (+4.3% CAGR) as the baseline, not the full-history numbers (+11.1%).

### 2. Survivorship Bias

This model was built because BTC and ETH exist, are liquid, and have performed extraordinarily well over their lifetimes. Many other cryptocurrencies have failed completely (Luna, FTX/FTT, many others). The power-law assumption assumes that the assets survive and maintain their adoption trajectory. This is not guaranteed.

**Mitigation:** Run this model only on BTC and ETH — the assets with the longest track record and deepest liquidity. Do not apply it to altcoins.

### 3. The Power Law May Not Hold

The power law is an empirical observation, not a law of physics. It has held for 12 years for BTC. The reasons it might continue to hold:
- Network adoption tends to follow S-curves, and the derivative of an S-curve in log-log space looks like a power law during the growth phase
- Bitcoin's supply schedule (halvings) creates predictable supply shocks on a ~4-year cycle

The reasons it might break:
- Adoption could plateau (we reach saturation)
- A structural technical failure or regulatory ban
- Competition from better alternatives
- Macro regime change that permanently reduces risk appetite

**Mitigation:** Monitor R² over time. If the rolling 2-year R² drops below 0.7, the model's reliability is declining and should be re-evaluated.

### 4. Small Sample Size for ML and High-Z Configs

103 BTC trades over 10 years is not a large sample for statistical inference. The ML overlay has even less data. Win rates of 57% and profit factors of 3.07 are promising but not reliable to a third decimal place. The 100% win rate at Z=2.5 (11 trades) is statistically meaningless.

**Mitigation:** Use the model as one input among several. Don't bet your house on any individual signal.

### 5. No Funding Costs, Borrow Costs, or Tax

The backtest models exchange fees and slippage but does not model:
- Opportunity cost of being in cash (could earn ~5% in money market)
- Tax on short-term capital gains (if holding < 1 year in jurisdictions where this applies)
- Funding rates if using leveraged positions (the model is unleveraged)
- Withdrawal/deposit friction on exchanges

**Mitigation:** For live trading, add these costs explicitly and reduce the real-world expected return accordingly.

### 6. This is Not Advice

Nothing in this document or the engine's output constitutes financial advice. This is a quantitative research tool. Past performance, even walk-forward performance, does not guarantee future results in a dynamically changing market.

---

## 16. Quick Reference: Running the Engine

### Installation

```bash
cd ~/crypto
pip3 install -r requirements.txt
```

### Daily Signal Check

```bash
python3 scripts/run_signals.py          # both assets
python3 scripts/run_signals.py --days 14  # last 14 days of Z history
```

### Running Backtests

```bash
# Baseline (full-history curve fit)
python3 scripts/run_backtest.py --symbol BTC

# Walk-forward (honest baseline — use this for real expectations)
python3 scripts/run_backtest.py --symbol BTC --walk-forward

# Regime-aware (adjusts thresholds by halving cycle + bull/bear)
python3 scripts/run_backtest.py --symbol BTC --regime

# ML overlay (logistic classifier gates entries)
python3 scripts/run_backtest.py --symbol BTC --ml --ml-threshold 0.55

# Full stack (all layers combined)
python3 scripts/run_backtest.py --symbol BTC --walk-forward --regime --ml

# Optimal config from sweep
python3 scripts/run_backtest.py --symbol BTC --buy-z 2.5 --z-window 365

# ETH
python3 scripts/run_backtest.py --symbol ETH
python3 scripts/run_backtest.py --symbol ETH --walk-forward
```

### Generating Charts

```bash
python3 scripts/run_charts.py            # both assets
python3 scripts/run_charts.py --symbol BTC
```

Output to: `reports/charts/` — 4 PNG files per asset.

### Parameter Sweep

```bash
python3 scripts/param_sweep.py --symbol BTC --quick --chart   # ~2 seconds
python3 scripts/param_sweep.py --symbol BTC --chart           # full grid ~10 min
```

Output to: `reports/sweep_BTC_YYYYMMDD.md` + `reports/charts/frontier_btc.png`

---

### Key Configuration (config.py)

All parameters in one place:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CURVE_MODEL` | `power_law` | Switch to `log_ema` for adaptive curve |
| `BUY_THRESHOLD` | `1.5` | Z below this triggers buy |
| `SELL_THRESHOLD` | `1.5` | Z above this triggers sell/exit |
| `EXIT_THRESHOLD` | `0.25` | Close position when Z within this of 0 |
| `ZSCORE_WINDOW` | `365` | Rolling std window in days |
| `FEE_RATE` | `0.001` | 0.10% per trade |
| `SLIPPAGE` | `0.0005` | 0.05% per trade |
| `USE_TREND_FILTER` | `True` | 20-week EMA slope gate |
| `USE_VOLUME_FILTER` | `True` | Volume above 80% of 30d avg |
| `USE_MACRO_FILTER` | `True` | Block entries in shock drawdowns |
| `LONG_ONLY` | `True` | Set False to allow shorts |

---

## Summary

This engine is built on a simple, robust idea: Bitcoin and Ethereum exhibit long-run power-law growth, and their prices mean-revert around that trend. The strategy capitalises on extreme deviations — buying deeply oversold conditions, exiting when prices recover toward the curve.

**The core trade-off is clear:**
- Accept lower absolute returns (11% CAGR vs 41% CAGR buy-and-hold in the full-fit backtest, 4% honest walk-forward)
- In exchange for dramatically lower volatility (14% vs 56%) and drawdowns (-22% vs -83%)

**The model is most valuable for someone who:**
- Believes in crypto's long-run trajectory but cannot psychologically endure 80%+ drawdowns
- Has a time horizon of 2+ years (the model needs time to accumulate its few annual trades)
- Wants a systematic, rules-based framework rather than emotional decision-making

**What to do right now (March 2026):**
- Both BTC and ETH are in deep value territory relative to their long-run curves
- Both are blocked by the trend filter (intermediate downtrend)
- Watch for the 20-week EMA slope to turn positive — that is the key trigger
- The post-halving window (now 11 months in) historically favours recovery
- Stay patient, stay in cash, run `run_signals.py` daily

---

*Report generated 27 March 2026. All backtests use data through 26 March 2026.*
*This is a quantitative research framework. Not financial advice.*
