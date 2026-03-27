# BTC Swing Trading Backtest — 2026-03-27

**Period:** 2016-01-01 → 2026-03-26
**Curve model:** power_law  ·  **Z-score window:** 365d  ·  **Buy threshold:** 2.0  ·  **Sell threshold:** 2.0
**Filters:** trend=OFF  volume=OFF  macro=OFF  ·  **Mode:** Long-Only
**Fees:** 0.10%  ·  **Slippage:** 0.05%  ·  **Starting capital:** $10,000

## Performance Summary

```
────────────────────────────────────────────────
  Metric               BTC Strategy   Buy & Hold
────────────────────────────────────────────────
  CAGR                       +18.2%       +40.7%
  Sharpe                       1.03         0.89
  Sortino                      0.65         1.20
  Calmar                       0.73         0.49
  Max Drawdown               -24.8%       -83.4%
  Ann. Volatility            +17.8%       +55.7%
  Win Rate                  +100.0%            —
  Profit Factor                 inf            —
  Avg Duration(d)              72.9            —
  Trades                         11            —
────────────────────────────────────────────────
  Period: 2016-01-01 → 2026-03-26
  BTC Strategy: $10,000 → $119,101  (+1091.0%)
  Buy & Hold: $10,000 → $1,583,842  (+15738.4%)
────────────────────────────────────────────────
```

## Trade Log

  #           Entry         Exit    Dir    Entry $     Exit $  Z entry  Z exit        P&L   P&L %
  ──── ──────────── ──────────── ────── ────────── ────────── ──────── ─────── ────────── ───────
  1      2016-01-13   2016-06-16   LONG        432        766    -2.10   -1.38     +9,802  +98.0%
  2      2016-06-21   2017-01-04   LONG        667      1,155    -2.44   -1.88    +19,463  +98.4%
  3      2017-01-05   2017-05-01   LONG      1,013      1,422    -3.17   -1.75    +17,605  +44.9%
  4      2020-03-12   2020-03-19   LONG      4,971      6,191    -2.92   -1.98    +12,920  +22.8%
  5      2020-03-22   2020-03-23   LONG      5,830      6,416    -2.21   -1.84     +1,433   +2.1%
  6      2020-03-29   2020-03-30   LONG      5,922      6,430    -2.16   -1.85       +962   +1.4%
  7      2023-05-08   2023-12-05   LONG     27,694     44,081    -2.14   -1.88    +41,370  +57.5%
  8      2023-12-11   2024-01-08   LONG     41,244     46,971    -2.42   -1.79     +9,708   +8.6%
  9      2024-01-12   2024-02-09   LONG     42,853     47,147    -2.61   -1.98     +8,585   +7.0%
  10     2025-11-14   2026-01-05   LONG     94,398     93,883    -2.24   -1.89     +6,240   +4.7%
  11     2026-01-07   2026-01-13   LONG     91,308     95,322    -2.09   -1.77       +531   +0.4%

## Entry Z-Score Distribution

```
  Z range         Trades   Win %    Avg P&L
  ────────────── ─────── ─────── ──────────
  [-10.0, -3.0)        1    100%    +17,605
  [-3.0, -2.0)       10    100%    +11,102
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
