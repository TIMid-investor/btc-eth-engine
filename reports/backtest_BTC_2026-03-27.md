# BTC Swing Trading Backtest — 2026-03-27

**Period:** 2016-01-01 → 2026-03-26
**Curve model:** power_law  ·  **Z-score window:** 365d  ·  **Buy threshold:** 1.5  ·  **Sell threshold:** 2.0
**Filters:** trend=ON  volume=ON  macro=ON  ·  **Mode:** Long-Only  ·  **Hold-Through-Cycle: ON**
**Fees:** 0.10%  ·  **Slippage:** 0.05%  ·  **Starting capital:** $10,000

## Performance Summary

```
────────────────────────────────────────────────
  Metric               BTC Strategy   Buy & Hold
────────────────────────────────────────────────
  CAGR                       +40.1%       +40.7%
  Sharpe                       1.17         0.89
  Sortino                      1.25         1.20
  Calmar                       0.81         0.49
  Max Drawdown               -49.7%       -83.4%
  Ann. Volatility            +33.8%       +55.7%
  Win Rate                  +100.0%            —
  Profit Factor                 inf            —
  Avg Duration(d)             428.5            —
  Trades                          2            —
────────────────────────────────────────────────
  Period: 2016-01-01 → 2026-03-26
  BTC Strategy: $10,000 → $1,493,548  (+14835.5%)
  Buy & Hold: $10,000 → $1,583,842  (+15738.4%)
────────────────────────────────────────────────
```

## Trade Log

  #           Entry         Exit    Dir    Entry $     Exit $  Z entry  Z exit        P&L   P&L %
  ──── ──────────── ──────────── ────── ────────── ────────── ──────── ─────── ────────── ───────
  1      2016-01-07   2017-08-14   LONG        458      4,325    -1.65    2.00    +84,157 +842.8%
  2      2020-03-20   2020-12-17   LONG      6,199     22,805    -1.97    2.20   +251,308 +267.3%

## Entry Z-Score Distribution

```
  Z range         Trades   Win %    Avg P&L
  ────────────── ─────── ─────── ──────────
  [-2.0, -1.5)        2    100%   +167,732
```

## Curve Fit Details

  price = 4.0931e-18 × days_since_genesis ^ 5.9247
  R² (log scale): 0.9208
  Today:  actual $      68,792   expected $     131,443   ratio 0.523

## Caveats

- Power-law exponent is fit on all available data (look-ahead in the curve itself). Use `--walk-forward` to eliminate this.
- Z-score uses a rolling trailing window — no look-ahead in signal generation.
- No tax, funding costs, or borrow fees for shorts modelled.
- Past performance of a mean-reversion model in a trending asset is not indicative of future results.
