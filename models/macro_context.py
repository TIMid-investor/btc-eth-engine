"""
models/macro_context.py — Broad macro environment context.

The crypto macro_filter in filters.py only detects BTC's own shock
drawdowns.  This module looks outward at the broader risk environment:

  SPY (S&P 500)   — broad market regime and drawdown
  VIX             — fear / volatility gauge
  DXY (dollar)    — dollar strength (inverse to crypto historically)
  Oil (WTI)       — inflationary pressure / stagflation risk
  BTC-SPY corr    — how much macro is currently driving crypto

Crypto correlation with equities is regime-dependent:
  Risk-ON bull    : BTC decouples, outperforms equities significantly
  Risk-OFF / bear : BTC correlates ~0.6–0.8 with SPY, falls with it
  Stagflation     : Worst case — no liquidity bid, dollar strong, rates high

A macro_score is computed (0 = neutral, positive = bearish for crypto)
and fed into the overall phase assessment in run_signals.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _YF = True
except ImportError:
    _YF = False

import os
DATA_CACHE = Path(os.environ.get("CRYPTO_DATA_DIR", str(Path.home() / "crypto-data")))

_MACRO_CACHE = DATA_CACHE / "macro_daily.parquet"
_SPY_CACHE   = DATA_CACHE / "spy_daily.parquet"


# ── Fetch ─────────────────────────────────────────────────────────────────────

def _fetch_ticker(ticker: str, start: str) -> pd.Series | None:
    """Download a single ticker close price; return None on failure."""
    if not _YF:
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, start=start, progress=False,
                             auto_adjust=True)
        if df.empty:
            return None
        close = df["Close"].squeeze()
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        close = close.dropna()
        return close if not close.empty else None
    except Exception:
        return None


def fetch_macro(days_back: int = 400) -> dict[str, pd.Series | None]:
    """
    Fetch macro price series.  Uses a daily parquet cache so repeated
    intra-day calls don't hit the network.

    Returns dict with keys: spy, vix, dxy, oil.
    Missing / failed tickers have value None.
    """
    start = (pd.Timestamp.today() - pd.DateOffset(days=days_back)).strftime("%Y-%m-%d")
    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Try cache first (valid for today)
    if _MACRO_CACHE.exists():
        try:
            cached = pd.read_parquet(_MACRO_CACHE)
            # Cache is valid if last row is from today or yesterday (market closed)
            last_dt = str(cached.index[-1].date())
            if last_dt >= (pd.Timestamp.today() - pd.DateOffset(days=1)).strftime("%Y-%m-%d"):
                result = {}
                for col in ["spy", "vix", "dxy", "oil"]:
                    if col in cached.columns:
                        s = cached[col].dropna()
                        result[col] = s if not s.empty else None
                    else:
                        result[col] = None
                return result
        except Exception:
            pass

    # Fetch from yfinance
    TICKERS = {
        "spy": "SPY",
        "vix": "^VIX",
        "dxy": "DX-Y.NYB",
        "oil": "CL=F",
    }
    result = {}
    frames = {}
    for key, ticker in TICKERS.items():
        s = _fetch_ticker(ticker, start)
        result[key] = s
        if s is not None:
            frames[key] = s

    # Write cache
    if frames:
        try:
            DATA_CACHE.mkdir(parents=True, exist_ok=True)
            df_cache = pd.DataFrame(frames)
            df_cache.index = pd.to_datetime(df_cache.index)
            df_cache.to_parquet(_MACRO_CACHE)
        except Exception:
            pass

    return result


def fetch_spy_for_corr(days_back: int = 120) -> pd.Series | None:
    """Fetch SPY (reuses macro cache if available)."""
    macro = fetch_macro(days_back=days_back)
    return macro.get("spy")


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_macro(macro: dict[str, pd.Series | None]) -> dict:
    """
    Compute macro environment signals.

    Returns dict with:
      spy_price     : latest SPY price
      spy_regime    : BULL / BEAR / NEUTRAL
      spy_vs_200d   : fraction above/below 200d SMA
      spy_drawdown  : fraction from 52-week high (negative = below)
      spy_30d_chg   : 30-day SPY return
      vix_level     : latest VIX
      vix_label     : EXTREME FEAR / ELEVATED / NORMAL / COMPLACENT
      dxy_price     : latest DXY level
      dxy_trend     : STRENGTHENING / WEAKENING / NEUTRAL
      oil_level     : latest WTI oil price
      oil_label     : HIGH (>$90) / ELEVATED ($70-90) / NORMAL (<$70)
      macro_score   : int, positive = more bearish for crypto
      risk_label    : RISK-OFF (severe) / RISK-OFF / MILDLY RISK-OFF /
                      NEUTRAL / RISK-ON
      signals       : list of (description, direction) for display
    """
    out = dict(
        spy_price=None, spy_regime="UNKNOWN", spy_vs_200d=None,
        spy_drawdown=None, spy_30d_chg=None,
        vix_level=None, vix_label="UNKNOWN",
        dxy_price=None, dxy_trend="UNKNOWN",
        oil_level=None, oil_label="UNKNOWN",
        macro_score=0, risk_label="UNKNOWN",
        signals=[],
    )
    score = 0
    signals = []

    # ── SPY ───────────────────────────────────────────────────────────────────
    spy = macro.get("spy")
    if spy is not None and len(spy) >= 10:
        spy_now = float(spy.iloc[-1])
        out["spy_price"] = spy_now

        # vs 200d SMA
        if len(spy) >= 200:
            sma200 = float(spy.rolling(200).mean().iloc[-1])
            vs200  = (spy_now - sma200) / sma200
            out["spy_vs_200d"] = vs200
            if vs200 < -0.08:
                out["spy_regime"] = "BEAR"
                score += 3
                signals.append((f"SPY {vs200*100:+.1f}% below 200d SMA — bear market", "bearish"))
            elif vs200 < -0.03:
                out["spy_regime"] = "CORRECTION"
                score += 2
                signals.append((f"SPY {vs200*100:+.1f}% below 200d SMA — correction", "bearish"))
            elif vs200 > 0.05:
                out["spy_regime"] = "BULL"
                score -= 1
                signals.append((f"SPY {vs200*100:+.1f}% above 200d SMA — bull market", "bullish"))
            else:
                out["spy_regime"] = "NEUTRAL"
                signals.append(f"SPY near 200d SMA ({vs200*100:+.1f}%) — transitioning")

        # Drawdown from 52-week high
        window = min(252, len(spy))
        high52 = float(spy.rolling(window, min_periods=20).max().iloc[-1])
        dd     = (spy_now - high52) / high52
        out["spy_drawdown"] = dd
        if dd < -0.20:
            score += 3
            signals.append((f"SPY {dd*100:.1f}% from 52w high — bear market territory", "bearish"))
        elif dd < -0.10:
            score += 2
            signals.append((f"SPY {dd*100:.1f}% from 52w high — significant correction", "bearish"))
        elif dd < -0.05:
            score += 1
            signals.append((f"SPY {dd*100:.1f}% from 52w high — pullback", "caution"))

        # 30-day change
        if len(spy) >= 30:
            chg30 = float(spy.iloc[-1] / spy.iloc[-30] - 1)
            out["spy_30d_chg"] = chg30
            if chg30 < -0.08:
                score += 1
                signals.append((f"SPY {chg30*100:+.1f}% in 30d — accelerating sell-off", "bearish"))

    # ── VIX ───────────────────────────────────────────────────────────────────
    vix = macro.get("vix")
    if vix is not None and len(vix) >= 3:
        vix_now = float(vix.iloc[-1])
        out["vix_level"] = vix_now
        if vix_now > 40:
            out["vix_label"] = "EXTREME FEAR"
            score += 3
            signals.append((f"VIX {vix_now:.1f} — extreme fear / potential capitulation", "extreme"))
        elif vix_now > 30:
            out["vix_label"] = "HIGH FEAR"
            score += 2
            signals.append((f"VIX {vix_now:.1f} — high fear", "bearish"))
        elif vix_now > 22:
            out["vix_label"] = "ELEVATED"
            score += 1
            signals.append((f"VIX {vix_now:.1f} — elevated uncertainty", "caution"))
        elif vix_now < 14:
            out["vix_label"] = "COMPLACENT"
            score -= 1
            signals.append((f"VIX {vix_now:.1f} — low fear / complacency", "bullish"))
        else:
            out["vix_label"] = "NORMAL"

    # ── DXY (dollar) ──────────────────────────────────────────────────────────
    dxy = macro.get("dxy")
    if dxy is not None and len(dxy) >= 30:
        dxy_now  = float(dxy.iloc[-1])
        out["dxy_price"] = dxy_now
        ema20    = float(dxy.ewm(span=20, adjust=False).mean().iloc[-1])
        ema60    = float(dxy.ewm(span=60, adjust=False).mean().iloc[-1])
        dxy_30d  = float(dxy.iloc[-1] / dxy.iloc[-30] - 1) if len(dxy) >= 30 else 0.0

        if dxy_now > ema20 and ema20 > ema60:
            out["dxy_trend"] = "STRENGTHENING"
            score += 2
            signals.append((f"DXY {dxy_now:.1f} — dollar strengthening (headwind for crypto)", "bearish"))
        elif dxy_now < ema20 and ema20 < ema60:
            out["dxy_trend"] = "WEAKENING"
            score -= 1
            signals.append((f"DXY {dxy_now:.1f} — dollar weakening (tailwind for crypto)", "bullish"))
        else:
            out["dxy_trend"] = "NEUTRAL"

    # ── Oil ───────────────────────────────────────────────────────────────────
    oil = macro.get("oil")
    if oil is not None and len(oil) >= 5:
        oil_now = float(oil.iloc[-1])
        out["oil_level"] = oil_now
        oil_30d = float(oil.iloc[-1] / oil.iloc[-30] - 1) if len(oil) >= 30 else 0.0

        if oil_now > 100:
            out["oil_label"] = "STAGFLATION RISK"
            score += 2
            signals.append((f"WTI ${oil_now:.0f} — above $100, stagflation pressure", "bearish"))
        elif oil_now > 85:
            out["oil_label"] = "ELEVATED"
            score += 1
            signals.append((f"WTI ${oil_now:.0f} — elevated, inflationary", "caution"))
        else:
            out["oil_label"] = "CONTAINED"

        # Crash signal (demand destruction / recession)
        if oil_30d < -0.15:
            score += 1
            signals.append((f"Oil {oil_30d*100:+.0f}% in 30d — demand destruction signal", "caution"))

    # ── Final risk label ──────────────────────────────────────────────────────
    out["macro_score"] = score
    out["signals"] = signals

    if score >= 7:
        out["risk_label"] = "RISK-OFF (severe)"
    elif score >= 4:
        out["risk_label"] = "RISK-OFF"
    elif score >= 2:
        out["risk_label"] = "MILDLY RISK-OFF"
    elif score <= -2:
        out["risk_label"] = "RISK-ON"
    elif score <= 0:
        out["risk_label"] = "NEUTRAL"
    else:
        out["risk_label"] = "MILDLY RISK-OFF"

    return out


def btc_spy_correlation(btc_prices: pd.Series,
                        spy: pd.Series | None,
                        window: int = 60) -> float | None:
    """
    Rolling 60-day BTC-SPY daily return correlation.

    High correlation (>0.6) means macro is closely driving crypto right now.
    Low / negative correlation means crypto is decoupling (cycle-driven).
    """
    if spy is None or len(spy) < window:
        return None
    try:
        btc_ret = btc_prices.pct_change().dropna()
        spy_ret = spy.pct_change().dropna()
        common  = btc_ret.index.intersection(spy_ret.index)
        if len(common) < window:
            return None
        return float(
            btc_ret.loc[common].tail(window).corr(spy_ret.loc[common].tail(window))
        )
    except Exception:
        return None
