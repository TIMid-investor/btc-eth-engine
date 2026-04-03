"""
backtest/tax.py — After-tax return modeling with FIFO lot tracking.

Models US capital gains tax on closed trades:
  - Short-term (held < 365 days): taxed at SHORT_TERM_RATE
  - Long-term  (held ≥ 365 days): taxed at LONG_TERM_RATE

Losses offset gains within the same year.  Remaining losses can be
carried forward (simplified — no wash-sale rules applied).

FIFO (first-in, first-out) is the default US lot disposal method.

Usage
-----
    from backtest.tax import apply_taxes, TaxLedger

    after_tax_equity, tax_log = apply_taxes(
        equity, trades, initial_capital=10_000,
        short_term_rate=0.37, long_term_rate=0.20
    )
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class TaxLot:
    """A single purchased lot of cryptocurrency."""
    buy_date:   pd.Timestamp
    buy_price:  float
    quantity:   float          # units of BTC/ETH


@dataclass
class TaxEvent:
    """A realized gain/loss event from selling one or more lots."""
    sell_date:      pd.Timestamp
    buy_date:       pd.Timestamp
    quantity:       float
    cost_basis:     float      # total cost for this lot
    proceeds:       float      # total proceeds for this lot
    gross_gain:     float      # proceeds - cost_basis
    holding_days:   int
    is_long_term:   bool
    tax_rate:       float
    tax_owed:       float      # max(0, gross_gain × tax_rate)
    net_gain:       float      # gross_gain - tax_owed


# ── FIFO lot ledger ────────────────────────────────────────────────────────────

class TaxLedger:
    """
    Tracks cost basis using FIFO lot disposal.

    Typical workflow per backtest trade:
        ledger = TaxLedger(short_term_rate=0.37, long_term_rate=0.20)
        ledger.buy(date, quantity, price)
        events = ledger.sell(date, quantity, price)
    """

    def __init__(
        self,
        short_term_rate: float = 0.37,
        long_term_rate:  float = 0.20,
        long_term_days:  int   = 365,
    ) -> None:
        self.short_term_rate = short_term_rate
        self.long_term_rate  = long_term_rate
        self.long_term_days  = long_term_days
        self._lots: list[TaxLot] = []    # FIFO queue
        self.events: list[TaxEvent] = []

    def buy(self, date: pd.Timestamp, quantity: float, price: float) -> None:
        """Record a purchase lot."""
        if quantity > 0:
            self._lots.append(TaxLot(
                buy_date=date,
                buy_price=price,
                quantity=quantity,
            ))

    def sell(
        self, date: pd.Timestamp, quantity: float, price: float
    ) -> list[TaxEvent]:
        """
        Dispose of lots FIFO, generate TaxEvents for each lot consumed.

        Returns the list of TaxEvents created in this sale.
        """
        remaining = quantity
        new_events: list[TaxEvent] = []

        while remaining > 1e-10 and self._lots:
            lot = self._lots[0]

            if lot.quantity <= remaining:
                # Consume this lot entirely
                consume_qty = lot.quantity
                self._lots.pop(0)
            else:
                # Partially consume this lot
                consume_qty = remaining
                lot.quantity -= consume_qty

            cost_basis = consume_qty * lot.buy_price
            proceeds   = consume_qty * price
            gross_gain = proceeds - cost_basis
            days_held  = (date - lot.buy_date).days
            lt         = days_held >= self.long_term_days
            rate       = self.long_term_rate if lt else self.short_term_rate
            tax        = max(0.0, gross_gain * rate)   # no tax on losses
            net_gain   = gross_gain - tax

            ev = TaxEvent(
                sell_date=date,
                buy_date=lot.buy_date,
                quantity=consume_qty,
                cost_basis=cost_basis,
                proceeds=proceeds,
                gross_gain=gross_gain,
                holding_days=days_held,
                is_long_term=lt,
                tax_rate=rate,
                tax_owed=tax,
                net_gain=net_gain,
            )
            new_events.append(ev)
            self.events.append(ev)
            remaining -= consume_qty

        return new_events

    @property
    def total_tax_owed(self) -> float:
        return sum(e.tax_owed for e in self.events)

    @property
    def total_gross_gain(self) -> float:
        return sum(e.gross_gain for e in self.events)

    @property
    def total_net_gain(self) -> float:
        return sum(e.net_gain for e in self.events)


# ── Main function: apply taxes to backtest results ─────────────────────────────

def apply_taxes(
    equity: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float = 10_000.0,
    short_term_rate: float = 0.37,
    long_term_rate:  float = 0.20,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Compute after-tax equity curve by deducting taxes on each closed trade.

    Approach
    --------
    For each closed trade the gain is computed from entry/exit prices.
    Tax is owed on profitable trades based on holding period (FIFO lots).
    Tax payments reduce the available capital on the trade exit date.
    The after-tax equity curve is then computed by scaling the original
    daily equity curve down by the cumulative tax drag.

    Parameters
    ----------
    equity          : pre-tax daily equity curve from run_backtest()
    trades          : trades DataFrame from run_backtest()
    initial_capital : starting capital (for buy-and-hold comparison)
    short_term_rate : federal + state short-term rate (default 37%)
    long_term_rate  : federal + state long-term rate (default 20%)

    Returns
    -------
    after_tax_equity : pd.Series — daily after-tax portfolio value
    tax_log          : pd.DataFrame — one row per tax event with all details
    """
    if trades.empty or "entry_date" not in trades.columns:
        # No trades → no tax drag
        return equity.copy(), pd.DataFrame()

    ledger = TaxLedger(
        short_term_rate=short_term_rate,
        long_term_rate=long_term_rate,
    )

    tax_by_date: dict[pd.Timestamp, float] = {}

    for _, trade in trades.iterrows():
        entry_date  = pd.Timestamp(trade["entry_date"])
        exit_date   = pd.Timestamp(trade["exit_date"])
        entry_price = float(trade.get("entry_price", 0) or 0)
        exit_price  = float(trade.get("exit_price",  0) or 0)
        direction   = str(trade.get("direction", "LONG"))

        if entry_price <= 0 or exit_price <= 0:
            continue

        if direction == "LONG":
            # Simulate buying 1 unit at entry, selling 1 unit at exit
            ledger.buy(entry_date, quantity=1.0, price=entry_price)
            events = ledger.sell(exit_date, quantity=1.0, price=exit_price)
            for ev in events:
                tax_by_date[exit_date] = tax_by_date.get(exit_date, 0.0) + ev.tax_owed
        # SHORT positions: tax on gains is identical in structure;
        # for simplicity we treat them symmetrically (short-term always, no lot tracking)
        elif direction == "SHORT":
            gross_gain = (entry_price - exit_price)   # profit if price fell
            if gross_gain > 0:
                tax = gross_gain * short_term_rate     # shorts always short-term
                tax_by_date[exit_date] = tax_by_date.get(exit_date, 0.0) + tax

    # Build after-tax equity curve
    # Strategy: subtract cumulative taxes from equity on each tax payment date
    after_tax = equity.copy()
    cumulative_tax = 0.0

    for dt in sorted(tax_by_date):
        if dt in after_tax.index:
            tax_amount = tax_by_date[dt]
            cumulative_tax += tax_amount
            # From this date forward, equity is reduced by this tax payment
            after_tax.loc[after_tax.index >= dt] -= tax_amount

    # Clip to 0 (can't go negative from taxes alone)
    after_tax = after_tax.clip(lower=0.0)

    # Build tax log DataFrame
    if ledger.events:
        tax_log = pd.DataFrame([{
            "sell_date":    e.sell_date.date(),
            "buy_date":     e.buy_date.date(),
            "holding_days": e.holding_days,
            "term":         "Long" if e.is_long_term else "Short",
            "tax_rate":     e.tax_rate,
            "gross_gain":   e.gross_gain,
            "tax_owed":     e.tax_owed,
            "net_gain":     e.net_gain,
        } for e in ledger.events])
    else:
        tax_log = pd.DataFrame()

    return after_tax, tax_log


# ── Summary comparison ─────────────────────────────────────────────────────────

def tax_summary(
    equity: pd.Series,
    after_tax_equity: pd.Series,
    bah_equity: pd.Series,
    tax_log: pd.DataFrame,
    initial_capital: float,
    short_term_rate: float = 0.37,
    long_term_rate: float  = 0.20,
) -> str:
    """
    Format a before/after-tax comparison table.

    Also computes after-tax buy-and-hold (assumes single entry at start,
    single exit at end → long-term rate if held > 365 days).
    """
    from backtest.metrics import cagr as _cagr, max_drawdown as _mdd, sharpe_ratio as _sharpe

    lines: list[str] = []
    A = lines.append

    A("=" * 68)
    A(f"  TAX ANALYSIS  (short-term: {short_term_rate*100:.0f}%  "
      f"long-term: {long_term_rate*100:.0f}%)")
    A("=" * 68)

    # Buy-and-hold after tax (1 entry, 1 exit)
    bah_days   = (bah_equity.index[-1] - bah_equity.index[0]).days
    bah_gain   = float(bah_equity.iloc[-1]) - initial_capital
    bah_rate   = long_term_rate if bah_days >= 365 else short_term_rate
    bah_tax    = max(0.0, bah_gain * bah_rate)
    bah_net    = float(bah_equity.iloc[-1]) - bah_tax
    bah_at_cagr = (bah_net / initial_capital) ** (365.25 / max(bah_days, 1)) - 1.0

    strat_cagr    = _cagr(equity)
    strat_at_cagr = _cagr(after_tax_equity)

    A(f"  {'Metric':<26} {'Pre-tax':>12} {'After-tax':>12}")
    A("  " + "─" * 50)
    A(f"  {'Strategy CAGR':<26} {strat_cagr*100:>11.1f}% {strat_at_cagr*100:>11.1f}%")
    A(f"  {'Strategy Max DD':<26} {_mdd(equity)*100:>11.1f}% {_mdd(after_tax_equity)*100:>11.1f}%")
    A(f"  {'Strategy Sharpe':<26} {_sharpe(equity):>12.2f} {_sharpe(after_tax_equity):>12.2f}")
    A(f"  {'Strategy Final ($)':<26} ${equity.iloc[-1]:>10,.0f} ${after_tax_equity.iloc[-1]:>10,.0f}")
    A("  " + "─" * 50)
    A(f"  {'B&H CAGR':<26} {_cagr(bah_equity)*100:>11.1f}% {bah_at_cagr*100:>11.1f}%")
    A(f"  {'B&H Final ($)':<26} ${bah_equity.iloc[-1]:>10,.0f} ${bah_net:>10,.0f}")
    A("  " + "─" * 50)

    if not tax_log.empty:
        st_trades = (tax_log["term"] == "Short").sum()
        lt_trades = (tax_log["term"] == "Long").sum()
        total_tax = tax_log["tax_owed"].sum()
        gross     = tax_log["gross_gain"].sum()
        A(f"  Short-term trades:    {st_trades:>4}  (rate {short_term_rate*100:.0f}%)")
        A(f"  Long-term trades:     {lt_trades:>4}  (rate {long_term_rate*100:.0f}%)")
        A(f"  Total gross gains:  ${gross:>10,.0f}")
        A(f"  Total tax owed:     ${total_tax:>10,.0f}")
        A(f"  Tax drag on capital: {total_tax/initial_capital*100:.1f}%")

    A("=" * 68)
    return "\n".join(lines)
