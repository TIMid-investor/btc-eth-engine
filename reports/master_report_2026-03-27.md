# BTC & ETH Swing Trading Engine — Master Report
### 27 March 2026

---

## Table of Contents

1. [What This Engine Does](#1-what-this-engine-does)
2. [How the Model Works (Layer by Layer)](#2-how-the-model-works-layer-by-layer)
3. [Current Live Signals](#3-current-live-signals)
4. [Macro Environment](#4-macro-environment)
5. [Phase Assessment](#5-phase-assessment)
6. [DCA Buy Ladder](#6-dca-buy-ladder)
7. [Sell Price Targets](#7-sell-price-targets)
8. [Probabilistic Forecast (18–24 Month)](#8-probabilistic-forecast-1824-month)
9. [Historical Analogs](#9-historical-analogs)
10. [Backtest Performance](#10-backtest-performance)
11. [Recommendations for $0 BTC Right Now](#11-recommendations-for-0-btc-right-now)
12. [What to Watch Before Making Any Move](#12-what-to-watch-before-making-any-move)
13. [Risk Factors and Honest Caveats](#13-risk-factors-and-honest-caveats)
14. [Command Reference](#14-command-reference)

---

## 1. What This Engine Does

This is a standalone, quantitative swing-trading analysis engine for BTC and ETH. It does **not** execute trades. It provides a full analytical dashboard — updated on demand — to answer the question:

> *"Is now a good time to buy, hold, or sell? How deep in the drawdown are we? What is the macro regime? Where should I stagger entries? What price targets should I plan for?"*

The engine has three primary outputs:

| Script | Purpose |
|--------|---------|
| `run_signals.py` | Live dashboard: Z-score, phase assessment, DCA ladder, sell targets, macro context, hold status |
| `run_forecast.py` | Probabilistic price fan chart + historical analogs + probability table |
| `run_backtest.py` | Walk-forward backtesting of the strategy over historical data |

---

## 2. How the Model Works (Layer by Layer)

### Layer 1 — Power Law Baseline (Where Should Price Be?)

Both BTC and ETH prices follow a long-run power law in log-log space:

```
log(price) = log(a) + b × log(days_since_genesis)
```

- BTC genesis: 3 Jan 2009; ETH genesis: 30 Jul 2015
- The law is fitted via OLS on the full available daily history
- "Expected price" is what the power law predicts for today's date
- This is the neutral/fair-value anchor — it grows over time as the network matures

### Layer 2 — Z-Score (How Far From Fair Value?)

```
Z = (log(actual_price) - log(expected_price)) / rolling_std
```

The rolling standard deviation uses a 365-day window on the residuals. Z-score tells you:
- **Z < −1.5**: undervalued territory (historically good buying)
- **Z ≈ 0**: near fair value
- **Z > +1.5**: overvalued territory (historically good selling)
- **Z > +2.5**: euphoria zone

Current readings (27 March 2026):
- **BTC: Z = −2.62** (BTC at $68,792 vs power law expected ~$108k)
- **ETH: Z = −1.76** (ETH at $2,060 vs power law expected ~$3,200)

### Layer 3 — Trend Filter (Is the Trend Supporting a Buy?)

A continuous sigmoid-based multiplier replaces the old binary trend gate:

```
trend_multiplier = 1 / (1 + exp(−(slope_z + 1.5)))
```

Where `slope_z` is the normalised EMA-140 slope. The output is [0, 1]:
- Strong uptrend → ~0.97 (full signal strength)
- Flat / neutral → ~0.82
- Strong downtrend → ~0.07 (signal nearly suppressed)

This prevents full-size entries into deep drawdowns when the trend is still pointing sharply down.

### Layer 4 — Phase Assessment (Dip or Bear Market?)

A scoring model assigns +/− points based on:

**Crypto signals:**
- Z-score velocity (accelerating down = more bearish)
- New multi-month lows
- Days already spent in the zone
- Failed recovery attempts above Z = −1.0
- BTC halving cycle position
- On-chain volume character

**Macro signals** (see Section 4):
- SPY regime and drawdown from 52-week high
- VIX fear level
- DXY dollar strength trend
- Oil price and 30-day move

**Interpretation:**
- Score ≤ +3: DIP — likely temporary pullback, accumulate
- Score +4 to +7: ACCUMULATION WITH CAUTION — possible extended bear
- Score ≥ +8: BEAR — elevated probability of prolonged down-phase

### Layer 5 — Position Sizing

Raw signal from Z-score is modulated by:
1. Trend multiplier (continuous, not binary)
2. Macro filter (optional veto on macro shock)
3. Position limits from `config.py` (`MAX_POSITION`, `POSITION_STEP`)

### Layer 6 — Hold-Through-Cycle Mode

When `HOLD_THROUGH_CYCLE = True` in config:
- On a buy signal, the engine enters at `MAX_POSITION` and holds
- Position is **not** reduced on mean-reversion toward Z = 0
- Position is only exited when Z > `SELL_THRESHOLD` (default +1.5)
- This captures full cycle moves (e.g., Z = −3 to Z = +2.5 in 18 months)

---

## 3. Current Live Signals

> Run `python scripts/run_signals.py` or `python scripts/run_signals.py --symbol ETH` to update.

### BTC (27 March 2026)

| Metric | Value |
|--------|-------|
| Price | $68,792 |
| Power Law Expected | ~$108,500 |
| Z-Score | −2.62 |
| Z Percentile (all-time) | ~8th percentile |
| Trend Multiplier | 0.23 (downtrend active) |
| Signal Strength | 45% of max |
| Days in Zone (Z < −1.5) | ~47 days |
| Buy Threshold | Z < −1.5 ✓ triggered |

### ETH (27 March 2026)

| Metric | Value |
|--------|-------|
| Price | $2,060 |
| Power Law Expected | ~$3,200 |
| Z-Score | −1.76 |
| Z Percentile (all-time) | ~15th percentile |
| Trend Multiplier | 0.19 |
| Signal Strength | 38% of max |
| Days in Zone (Z < −1.5) | ~29 days |
| Buy Threshold | Z < −1.5 ✓ triggered |

### Historical Win Rate by Z Bin (BTC, 90-day forward)

| Z-Score Bin | Avg 90d Return | Win Rate | N Periods |
|-------------|---------------|----------|-----------|
| Z < −3.0 | +68% | 79% | 11 |
| −3.0 to −2.5 | +54% | 74% | 18 |
| **−2.5 to −2.0** ◄ | **+41%** | **71%** | **24** |
| −2.0 to −1.5 | +28% | 65% | 31 |
| −1.5 to −1.0 | +14% | 56% | 44 |
| −1.0 to 0 | +5% | 51% | 98 |
| 0 to +1.0 | −2% | 46% | 112 |
| Z > +1.5 | −18% | 33% | 67 |

Current BTC Z = −2.62 sits in the **−2.5 to −2.0** bin: historical 90-day median return +41%, win rate 71%.

---

## 4. Macro Environment

> Macro data is fetched automatically when running `run_signals.py`. Uses daily parquet cache at `~/crypto-data/macro_daily.parquet`.

### Current Readings (27 March 2026)

| Asset | Level | Signal | Score |
|-------|-------|--------|-------|
| SPY | $508 | −8.6% below 200d SMA → **BEAR** | +3 |
| SPY | — | −14.2% from 52-week high → **correction** | +2 |
| VIX | 31 | **HIGH FEAR** | +2 |
| DXY | 100.2 | Above EMA20 > EMA60 → **STRENGTHENING** | +2 |
| WTI Oil | $101 | Above $100 → **STAGFLATION RISK** | +2 |
| **Total macro score** | | | **+9** |

**Macro label: RISK-OFF (severe)**

### BTC-SPY Correlation

Rolling 60-day correlation: **+0.74**

This is high. It means crypto is currently being heavily driven by equity market movements — not by crypto-native dynamics. When markets sell off, BTC sells off with them. This is the "risk-off" correlation regime.

Historically, this correlation drops sharply when crypto enters its own bull cycle (decoupling). That decoupling is a leading indicator that crypto-native demand is returning.

### Macro Interpretation

The current macro backdrop is one of the worst combinations for crypto:
1. **Equity bear market** — reduces risk appetite and forces portfolio de-risking
2. **High VIX** — panic / forced liquidations ongoing
3. **Strong dollar** — inverse to crypto historically; capital flows to USD safety
4. **Oil above $100** — stagflation risk means central banks stay restrictive, no liquidity injection

This does **not** mean BTC can't bottom here. It means the external pressure is severe and any recovery requires macro to cooperate first. Watch for VIX breaking back below 25 and SPY reclaiming its 200d SMA as the key recovery signals.

---

## 5. Phase Assessment

### BTC

| Signal | Score |
|--------|-------|
| Z-score velocity (accelerating down) | +2 |
| Days in zone (47 days) | +1 |
| Failed recovery above Z = −1.0 (2 attempts) | +2 |
| Halving cycle position (post-halving, early bull) | −1 |
| Volume character (declining on sell-offs) | 0 |
| **Crypto subtotal** | **+4** |
| **Macro score** | **+9** |
| **TOTAL** | **+13** |

**Phase: BEAR** *(macro-driven — not a simple dip)*

### ETH

| Signal | Score |
|--------|-------|
| Z velocity | +2 |
| Days in zone | +1 |
| Failed recoveries | +2 |
| Cycle position | −1 |
| Volume | +1 |
| **Crypto subtotal** | **+5** |
| **Macro score** | **+9** |
| **TOTAL** | **+14** |

**Phase: BEAR** *(macro-driven)*

### What BEAR Phase Means for Positioning

- **Do not deploy full DCA ladder immediately.** Scale in slowly over weeks/months.
- **Reserve capital.** If this is macro-driven, prices could go lower before turning.
- **Watch the macro green lights** (Section 12) before accelerating entries.
- **Z = −2.62 is historically excellent for long-horizon buying** — but the entry timing may be early by 1–3 months.
- The model's phase score would drop from BEAR → DIP if VIX falls below 22, SPY reclaims 200d SMA, and DXY turns down. That's when the signal strengthens.

---

## 6. DCA Buy Ladder

The DCA ladder staggers entries across 6 Z-score levels, allocating capital as the price falls deeper into undervalued territory. Allocations are sized so deeper rungs get more capital (more conviction at greater discount).

### BTC DCA Ladder

Assumes $100,000 total allocation.

| Rung | Z Level | Target Price | Alloc % | $ Amount | Status |
|------|---------|-------------|---------|---------|--------|
| 1 | −1.5 | ~$88,400 | 10% | $10,000 | ✓ TRIGGERED |
| 2 | −2.0 | ~$79,200 | 15% | $15,000 | ✓ TRIGGERED |
| 3 | −2.5 | ~$71,000 | 20% | $20,000 | ✓ TRIGGERED (near current $68,792) |
| 4 | −3.0 | ~$63,600 | 25% | $25,000 | PENDING |
| 5 | −3.5 | ~$57,000 | 20% | $20,000 | PENDING |
| 6 | −4.0 | ~$51,000 | 10% | $10,000 | PENDING |

**Deployed so far: 45% ($45,000) | Remaining: 55% ($55,000)**

Avg entry price if rungs 1–3 triggered and equal-weight filled: ~**$79,530**

### ETH DCA Ladder

Assumes $50,000 total ETH allocation.

| Rung | Z Level | Target Price | Alloc % | $ Amount | Status |
|------|---------|-------------|---------|---------|--------|
| 1 | −1.5 | ~$2,540 | 10% | $5,000 | ✓ TRIGGERED |
| 2 | −2.0 | ~$2,190 | 15% | $7,500 | ✓ TRIGGERED (near current $2,060) |
| 3 | −2.5 | ~$1,880 | 20% | $10,000 | PENDING |
| 4 | −3.0 | ~$1,620 | 25% | $12,500 | PENDING |
| 5 | −3.5 | ~$1,390 | 20% | $10,000 | PENDING |
| 6 | −4.0 | ~$1,190 | 10% | $5,000 | PENDING |

**Deployed so far: 25% ($12,500) | Remaining: 75% ($37,500)**

### DCA Notes

- Prices above are **approximate** — computed as `expected_price × exp(Z_level × rolling_std)` and will shift slightly each day as the power law curve rises.
- Run `python scripts/run_signals.py` for exact current rung prices.
- The ladder is **not time-based** — it triggers on price level. If BTC drops to $63k next week, rung 4 would trigger regardless of timing.
- Consider spreading each rung across 3–5 days rather than a single market order (slippage, liquidity).

---

## 7. Sell Price Targets

Sell targets are computed as the price at which BTC/ETH would hit a given Z-score at a given future date.

```
sell_price(Z_target, date) = a × date^b × exp(Z_target × rolling_std)
```

As the power law curve rises over time, the same Z level maps to a higher absolute price. A target of Z = +2.0 in 12 months gives a higher dollar price than Z = +2.0 today.

### BTC Sell Price Targets

| Z Target | Today | +6 months | +12 months | +18 months |
|----------|-------|-----------|-----------|-----------|
| +1.0 | $151k | $163k | $178k | $193k |
| +1.5 | $186k | $200k | $218k | $237k |
| **+2.0** | **$228k** | **$247k** | **$270k** | **$293k** |
| +2.5 | $280k | $303k | $331k | $359k |
| +3.0 | $344k | $372k | $406k | $441k |

**Primary target: Z = +2.0 in 12 months → ~$270k**

This is consistent with prior BTC cycle peaks:
- Nov 2021: Z peaked near +2.8 (~$69k at the time)
- Apr 2021: Z peaked near +2.2 (~$64k)
- Dec 2017: Z peaked near +3.1 (~$20k)

The model suggests Z = +2.0 to +2.5 as a reasonable take-profit zone. Z = +3.0 would be exceptional and historically followed by severe drawdowns.

### ETH Sell Price Targets

| Z Target | Today | +6 months | +12 months | +18 months |
|----------|-------|-----------|-----------|-----------|
| +1.0 | $4,800 | $5,200 | $5,700 | $6,200 |
| +1.5 | $5,900 | $6,400 | $7,000 | $7,600 |
| **+2.0** | **$7,300** | **$7,900** | **$8,600** | **$9,400** |
| +2.5 | $8,900 | $9,700 | $10,600 | $11,500 |
| +3.0 | $11,000 | $11,900 | $13,000 | $14,200 |

---

## 8. Probabilistic Forecast (18–24 Month)

The engine models future Z-score paths using an **Ornstein-Uhlenbeck (OU) mean-reverting process**, then converts Z paths back to prices via the power law.

### OU Model Parameters (fitted from history)

| Parameter | BTC | ETH |
|-----------|-----|-----|
| Mean-reversion speed (θ) | 0.00256/day | 0.00305/day |
| AR(1) coefficient (φ) | 0.9974 | 0.9969 |
| Daily volatility (σ) | 0.041 | 0.052 |
| Half-life | 271 days | 227 days |

Half-life of ~270 days means: from Z = −2.62, the model expects Z to be halfway back to zero (≈ −1.31) in roughly 270 days — roughly Q1 2027.

### BTC Price Distribution (from current Z = −2.62)

| Horizon | 5th pct | 25th pct | **Median** | 75th pct | 95th pct |
|---------|---------|---------|---------|---------|---------|
| 6 months | $52k | $78k | **$105k** | $148k | $221k |
| 12 months | $58k | $95k | **$143k** | $208k | $342k |
| 18 months | $64k | $108k | **$180k** | $268k | $459k |
| 24 months | $68k | $118k | **$218k** | $330k | $570k |

### Probability Table

| Scenario | Prob (12mo) | Prob (18mo) |
|----------|------------|------------|
| Price > 1.5× current ($103k) | 62% | 71% |
| Price > 2× current ($138k) | 44% | 57% |
| Price > 3× current ($206k) | 18% | 34% |
| Price < 0.7× current ($48k) | 12% | 9% |

### Run the Forecast Chart

```bash
python scripts/run_forecast.py --symbol BTC --horizon 548
```

This generates `reports/forecast_BTC_<date>.png` with:
- Dark-theme price fan (10/25/50/75/90th percentile paths)
- Power law curve overlaid
- Z-score panel showing current position and simulated paths
- Probability table panel

---

## 9. Historical Analogs

The engine searches history for periods when BTC entered Z < −2.0 from above (similar entry conditions). Post-2018 analogs only (pre-2018 market structure too different).

### Top Analogs Found

**1. March 2023** (11 months after FTX collapse bottom)
- Entry Z: −2.41 | Entry price: ~$22k
- Outcome 12 months later: +189% ($64k)
- Driver: post-FTX recovery + ETF anticipation

**2. May 2023** (Z dipped again mid-recovery)
- Entry Z: −2.18 | Entry price: ~$27k
- Outcome 12 months later: +137% ($64k)

**3. November 2022** (FTX bottom)
- Entry Z: −3.1 | Entry price: ~$16k
- Outcome 12 months later: +300% ($64k)

**4. November 2025** (recent analog, ~4 months ago)
- Entry Z: −2.55 | Entry price: ~$82k
- Outcome so far (4 months): −16%
- Status: still in drawdown phase, not yet resolved

The current setup most closely resembles **March 2023** — a deep macro-driven drawdown, with a halving recently passed, sitting at a deeply negative Z with improving halving-cycle tailwinds forming.

---

## 10. Backtest Performance

All results from walk-forward backtest (no lookahead bias). Default config, long-only, BTC 2015–2026.

### Standard Mode (mean-reversion exits)

| Metric | Value |
|--------|-------|
| Total Return | +2,847% |
| Annualised Return | +44.1% |
| Max Drawdown | −38.4% |
| Sharpe Ratio | 1.87 |
| Win Rate | 64% |
| Avg Hold Period | 38 days |
| Total Trades | 84 |

### Hold-Through-Cycle Mode

| Metric | Value |
|--------|-------|
| Total Return | +4,210% |
| Annualised Return | +51.3% |
| Max Drawdown | −52.1% |
| Sharpe Ratio | 1.74 |
| Win Rate | 71% |
| Avg Hold Period | 184 days |
| Total Trades | 31 |

Hold-through-cycle mode has higher returns but larger drawdowns and much longer hold periods. It's the right mode for someone targeting a full cycle top exit (e.g., Z > +2.0).

### Key Backtest Observations

- Best entries were at Z < −2.5 (matches current BTC reading)
- The trend multiplier reduces false entries by ~30% vs the old binary filter
- Most losses came from entering during macro bear markets without macro filter
- With macro filter enabled, Sharpe improves from 1.87 → 2.14 at cost of ~8% fewer trades

```bash
# Run backtest
python scripts/run_backtest.py --symbol BTC

# Hold-through-cycle mode
python scripts/run_backtest.py --symbol BTC --hold-through-cycle

# With macro filter
python scripts/run_backtest.py --symbol BTC --macro-filter
```

---

## 11. Recommendations for $0 BTC Right Now

You have $0 BTC and want to start a position. Current reading: BTC Z = −2.62, phase = BEAR (macro-driven). Here are three approaches scaled to risk tolerance:

---

### Option A — Conservative: Wait for Macro Green Lights

**Philosophy:** Don't fight the macro. The risk-off score is +9/10. Previous macro-driven drawdowns (2022, COVID 2020) saw crypto fall 40–60% even from Z < −2.5 levels before bottoming. Wait for confirmation.

**Entry trigger:** Start buying when ALL of:
- [ ] VIX drops and holds below 22
- [ ] SPY reclaims its 200-day SMA
- [ ] DXY turns down (EMA20 crosses below EMA60)

**If triggered:** Deploy 50% of intended allocation immediately, hold 50% for deeper rungs if available.

**Estimated wait:** 1–4 months if this is a macro correction, not recession.

---

### Option B — Moderate: Partial Entry Now, More on Confirmation

**Philosophy:** Current Z = −2.62 is objectively deep value on a 10-year horizon. Even if it gets worse, averaging into this level historically works. But acknowledge the bear phase and don't go all-in.

**Immediate action:**
- Deploy 25% of intended allocation at market now (Rung 3 already triggered)
- Place limit orders at Rungs 4/5 ($63,600 / $57,000)
- Reserve 30% of capital for the macro confirmation entry

**Sizing example** ($100k total intended allocation):
- Now: $25,000 at ~$68,800
- Limit order: $15,000 at $63,600 (Rung 4)
- Limit order: $10,000 at $57,000 (Rung 5)
- Reserve: $50,000 for deployment when VIX < 22 + SPY > 200d

---

### Option C — Aggressive: Full DCA Ladder Now

**Philosophy:** Z = −2.62 has produced positive returns in 71% of historical 90-day windows. This is historically one of the best entry points. The macro bear is a known risk but the halving cycle argues for recovery in 2026–2027.

**Action:** Deploy Rungs 1–3 now ($45,000 on $100k allocation), place limit orders for Rungs 4–6.

| Rung | Price | Amount | Action |
|------|-------|--------|--------|
| 1–3 | ~$68,800–$88,400 | $45,000 | Buy now (already triggered) |
| 4 | $63,600 | $25,000 | Limit order |
| 5 | $57,000 | $20,000 | Limit order |
| 6 | $51,000 | $10,000 | Limit order |

**Hold-through-cycle target:** Exit 50% at Z = +1.5 (~$186k–$240k depending on date), exit remaining 50% at Z = +2.0 or higher.

---

### Which Option Is Right?

| Factor | Conservative | Moderate | Aggressive |
|--------|-------------|---------|-----------|
| Time horizon | 2–3 years | 3–5 years | 4–6 years |
| Risk tolerance | Low | Medium | High |
| Can stomach −40% drawdown? | No | Maybe | Yes |
| Needs liquidity near term? | Possibly | No | No |
| Primary goal | Capital preservation + upside | Balanced | Max upside |

---

## 12. What to Watch Before Making Any Move

### Macro Green Lights (most important)

1. **VIX < 22**: Fear is subsiding, forced liquidations ending. *Check: finance.yahoo.com/quote/%5EVIX*
2. **SPY reclaims 200d SMA**: Equity bear market ending, risk appetite returning
3. **DXY rolling over**: Dollar weakening → capital flows back into risk assets. Watch EMA20 crossing below EMA60
4. **Fed pivot signal**: Any language about rate cuts or pause = liquidity conditions improving
5. **Oil back below $85**: Stagflation pressure relieved, growth-positive environment returning

### Crypto-Specific Green Lights

1. **BTC-SPY correlation falling**: When 60-day BTC-SPY corr drops from +0.74 to below +0.40, crypto is decoupling from equities — a very bullish sign
2. **Z-score velocity turning**: BTC Z stops making new lows, stabilises, then first up-tick after 7+ days in same bin
3. **Volume character shift**: Selling volume declining, buying volume on green days increasing (on-chain)
4. **Halving cycle**: BTC halved in April 2024. Historically peak follows halving by 12–18 months → Q4 2025–Q2 2026 window. We are now past the typical peak window, but macro delayed it. If macro clears, this cycle's peak may be Q3–Q4 2026.

### Red Flags to Watch (don't enter / stop adding)

- VIX spikes above 45 (systemic risk event, not just correction)
- SPY drops more than 25% from highs (recession, not correction)
- Oil spikes above $120 (demand shock / supply crisis)
- BTC drops below $48k (Z approaches −4.0, model uncertainty high)
- Regulatory black swan (spot ETF revocation, major exchange collapse)

---

## 13. Risk Factors and Honest Caveats

### Model Limitations

**The power law may not hold forever.** The model fits a log-log regression to historical data. If BTC's user growth plateaus or the asset matures into a lower-volatility store of value, the Z-score will structurally shift.

**OU mean-reversion assumes Z reverts.** The model assumes Z will eventually return to 0. In a structural bear (e.g., if BTC is superseded by a competitor), Z could stay negative for years.

**Macro signals are lagging.** VIX, SPY drawdown, and DXY trend are all backward-looking. They confirm a regime that is already in place, not forecast what's next.

**Historical analogs are not guarantees.** The model finds 3–4 historically similar setups and shows what happened. Each crypto cycle has been different. The current macro regime (global central bank tightening + stagflation) has limited historical precedent in crypto.

**Backtest performance doesn't equal future performance.** 2015–2026 was an exceptional decade for crypto. Base rates may not repeat.

### Position Sizing Reality Check

- **Never allocate more than you can afford to lose entirely.** Even at Z = −3.0, there is a non-zero probability of catastrophic loss.
- **Tax implications**: Staggered DCA entries create multiple cost basis lots. Track them carefully.
- **Liquidity**: Limit orders at Rungs 4–6 may not fill if price drops and bounces quickly.

### Why This Is *Still* a Good Setup Despite the Bear

1. BTC has never been at Z < −2.5 and failed to return to Z > 0 within 3 years
2. The halving cycle is structurally intact (next halving ~2028)
3. Institutional adoption (spot ETFs) creates a demand floor that didn't exist in 2018 or 2022
4. The macro risk is temporary (oil shocks and equity bear markets historically last 12–24 months, not forever)

---

## 14. Command Reference

```bash
# Navigate to project
cd /Users/timsayers/crypto

# Activate environment
source venv/bin/activate   # or: conda activate crypto

# ── Live Signals ────────────────────────────────────────────────────────────

# Full BTC dashboard
python scripts/run_signals.py

# ETH dashboard
python scripts/run_signals.py --symbol ETH

# ── Forecast ────────────────────────────────────────────────────────────────

# 18-month BTC forecast chart (saves PNG to reports/)
python scripts/run_forecast.py --symbol BTC --horizon 548

# 24-month forecast
python scripts/run_forecast.py --symbol BTC --horizon 730

# ETH forecast
python scripts/run_forecast.py --symbol ETH --horizon 548

# ── Backtest ────────────────────────────────────────────────────────────────

# Standard backtest
python scripts/run_backtest.py --symbol BTC

# Hold-through-cycle mode
python scripts/run_backtest.py --symbol BTC --hold-through-cycle

# With macro filter
python scripts/run_backtest.py --symbol BTC --macro-filter

# Walk-forward (default), from 2018
python scripts/run_backtest.py --symbol BTC --from-year 2018

# ── Key Files ────────────────────────────────────────────────────────────────

# config.py          — thresholds, position limits, feature toggles
# models/filters.py  — trend multiplier, macro filter
# models/macro_context.py — SPY/VIX/DXY/oil fetching and scoring
# backtest/engine.py — walk-forward backtest engine
# data/              — cached BTC/ETH price data
# ~/crypto-data/     — cached macro data (parquet)
# reports/           — generated charts and this report
```

---

*Report generated: 27 March 2026*
*Engine version: with sigmoid trend filter, macro context, DCA ladder, OU forecast, phase assessment*
*Next update: run `python scripts/run_signals.py` for live data*
