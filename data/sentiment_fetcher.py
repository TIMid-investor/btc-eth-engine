"""
data/sentiment_fetcher.py — Market sentiment data from free public sources.

Two sources:

1. Crypto Fear & Greed Index (alternative.me)  ← Start here
   ✅ Free, no API key, full history since 2018
   Score 0–100:  0 = Extreme Fear, 100 = Extreme Greed
   As a demand signal: rising = more market participation
   As an exit signal:  >75 (greed zone) rolling over = exit warning

2. Reddit sentiment (PRAW + VADER)
   ✅ Free — requires Reddit API credentials (register at reddit.com/prefs/apps)
   Subreddits: r/Bitcoin, r/ethereum, r/CryptoCurrency
   Engagement-weighted daily sentiment score (-1 to +1)
   Set env vars: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

Reddit module call chain
------------------------
  data/sentiment_fetcher.py   ← THIS FILE — fetch + cache raw Reddit posts (PRAW)
      └─ models/reddit_sentiment.py   — VADER + FinBERT scoring of post text
      └─ models/reddit_narrative.py   — phase detection (capitulation→euphoria)
  scripts/run_reddit.py       — terminal dashboard consuming all three

  models/reddit_collector.py  — DEPRECATED predecessor to this file; do not use.

Usage
-----
    from data.sentiment_fetcher import fetch_fear_greed, fetch_reddit_sentiment

    fg  = fetch_fear_greed(start="2018-01-01")
    red = fetch_reddit_sentiment("bitcoin", start="2022-01-01")  # needs PRAW creds
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_CACHE
from data.cache_utils import (load_cache as _load_cache, save_cache as _save_cache,
                               filter_dates as _filter_dates, get_last_date as _get_last_date)

_SESSION = requests.Session()
_SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "btc-eth-engine/1.0",
})
_TIMEOUT = 20


# ══════════════════════════════════════════════════════════════════════════════
# Crypto Fear & Greed Index  (alternative.me — free, no key)
# ══════════════════════════════════════════════════════════════════════════════

_FG_URL = "https://api.alternative.me/fng/"


def fetch_fear_greed(
    start: str = "2018-02-01",   # index started Feb 2018
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch the Crypto Fear & Greed Index history (free, no key).

    Columns
    -------
    fear_greed       : raw score 0–100 (0 = Extreme Fear, 100 = Extreme Greed)
    fear_greed_label : text label (e.g. "Greed", "Extreme Fear")
    fear_greed_norm  : scaled to [-1, +1]  (Fear=-1, Neutral=0, Greed=+1)

    Parameters
    ----------
    start : ISO date string (earliest available: 2018-02-01)
    end   : ISO date string; defaults to today
    """
    end        = end or str(date.today())
    cache_path = DATA_CACHE / "fear_greed.parquet"

    cached   = None
    last_bar = _get_last_date(cache_path)
    if last_bar is None and not force_refresh:
        cached   = _load_cache(cache_path)
        last_bar = cached.index.max() if cached is not None else None
    if last_bar is not None and not force_refresh:
        if (pd.Timestamp(end) - last_bar).days <= 1:
            if cached is None:
                cached = _load_cache(cache_path)
            if cached is not None:
                return _filter_dates(cached, start, end)

    if cached is None:
        cached = _load_cache(cache_path)

    df = _download_fear_greed()
    if df is None or df.empty:
        if cached is not None:
            print("  [fear_greed] download failed; using cache", flush=True)
            return _filter_dates(cached, start, end)
        raise RuntimeError(
            "Could not fetch Fear & Greed Index from alternative.me"
        )

    _save_cache(df, cache_path)
    return _filter_dates(df, start, end)


def _download_fear_greed() -> pd.DataFrame | None:
    """Download full history from alternative.me."""
    params = {"limit": 0, "format": "json"}   # limit=0 = all history
    try:
        resp = _SESSION.get(_FG_URL, params=params, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception as exc:
        print(f"  [fear_greed] request error: {exc}", flush=True)
        return None

    rows = data.get("data", [])
    if not rows:
        return None

    dates   = pd.to_datetime([int(r["timestamp"]) for r in rows], unit="s").normalize()
    scores  = [int(r["value"]) for r in rows]
    labels  = [r["value_classification"] for r in rows]

    df = pd.DataFrame({
        "fear_greed":       scores,
        "fear_greed_label": labels,
    }, index=dates)
    df.index.name = "date"
    df.sort_index(inplace=True)

    # Normalise to [-1, +1]
    df["fear_greed_norm"] = (df["fear_greed"] - 50.0) / 50.0

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Reddit sentiment  (PRAW + VADER — free, requires Reddit app credentials)
# ══════════════════════════════════════════════════════════════════════════════

_SUBREDDITS = {
    "BTC": ["Bitcoin", "CryptoCurrency"],
    "ETH": ["ethereum", "CryptoCurrency"],
}


def fetch_reddit_sentiment(
    symbol: str = "BTC",
    start: str = "2020-01-01",
    end: str | None = None,
    force_refresh: bool = False,
    posts_per_day: int = 25,
) -> pd.DataFrame | None:
    """
    Fetch engagement-weighted daily Reddit sentiment (free, requires PRAW creds).

    Environment variables required:
        REDDIT_CLIENT_ID     — from reddit.com/prefs/apps
        REDDIT_CLIENT_SECRET — from reddit.com/prefs/apps
        REDDIT_USER_AGENT    — any string, e.g. "btc-eth-engine/1.0"

    Returns daily DataFrame with columns:
        reddit_sentiment      : engagement-weighted avg sentiment (-1 to +1)
        reddit_post_count     : number of posts sampled that day
        reddit_bullish_ratio  : fraction of posts with positive sentiment

    Returns None if credentials are missing (with helpful message).

    Note on historical data
    -----------------------
    PRAW only accesses recent posts (~1000 newest per subreddit).
    For backtesting history, this fetcher accumulates daily readings over time.
    Run it daily (e.g. via cron) to build up a historical record.
    """
    client_id     = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent    = os.environ.get("REDDIT_USER_AGENT", "btc-eth-engine/1.0")

    if not client_id or not client_secret:
        print(
            "  [reddit] REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set.\n"
            "  Register a free app at https://www.reddit.com/prefs/apps\n"
            "  Set env vars and re-run to enable Reddit sentiment.",
            flush=True,
        )
        return None

    end        = end or str(date.today())
    symbol     = symbol.upper()
    cache_path = DATA_CACHE / f"reddit_sentiment_{symbol}.parquet"

    cached   = None
    last_bar = _get_last_date(cache_path)
    if last_bar is None and not force_refresh:
        cached   = _load_cache(cache_path)
        last_bar = cached.index.max() if cached is not None else None
    if last_bar is not None and not force_refresh:
        if (pd.Timestamp(end) - last_bar).days <= 1:
            if cached is None:
                cached = _load_cache(cache_path)
            if cached is not None:
                return _filter_dates(cached, start, end)

    if cached is None:
        cached = _load_cache(cache_path)

    df = _download_reddit(symbol, client_id, client_secret, user_agent,
                          posts_per_day)
    if df is None or df.empty:
        if cached is not None:
            return _filter_dates(cached, start, end)
        return None

    # Merge with cache (accumulate historical readings)
    if cached is not None:
        combined = pd.concat([cached, df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        df = combined

    _save_cache(df, cache_path)
    return _filter_dates(df, start, end)


def _download_reddit(
    symbol: str,
    client_id: str,
    client_secret: str,
    user_agent: str,
    posts_per_day: int,
) -> pd.DataFrame | None:
    try:
        import praw  # type: ignore
    except ImportError:
        print(
            "  [reddit] praw not installed. Run: pip install praw>=7.7",
            flush=True,
        )
        return None

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
        analyzer = SentimentIntensityAnalyzer()
    except ImportError:
        print(
            "  [reddit] vaderSentiment not installed. Run: pip install vaderSentiment>=3.3\n"
            "  Falling back to simple keyword sentiment.",
            flush=True,
        )
        analyzer = None

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    subreddits = _SUBREDDITS.get(symbol, ["CryptoCurrency"])
    today      = pd.Timestamp(date.today())

    posts: list[dict] = []
    for sub_name in subreddits:
        try:
            sub = reddit.subreddit(sub_name)
            for submission in sub.hot(limit=posts_per_day):
                text = submission.title + " " + (submission.selftext or "")
                score = _score_text(text, analyzer)
                posts.append({
                    "date":      today,
                    "score":     submission.score,
                    "comments":  submission.num_comments,
                    "sentiment": score,
                })
        except Exception as exc:
            print(f"  [reddit] r/{sub_name} error: {exc}", flush=True)

    if not posts:
        return None

    df = pd.DataFrame(posts)
    df["engagement"] = df["score"] + df["comments"] + 1   # +1 avoids zero weight

    # Engagement-weighted average sentiment
    today_df = df.groupby("date").apply(
        lambda g: pd.Series({
            "reddit_sentiment":     np.average(g["sentiment"], weights=g["engagement"]),
            "reddit_post_count":    len(g),
            "reddit_bullish_ratio": (g["sentiment"] > 0).mean(),
        })
    ).reset_index(level=0, drop=True)

    today_df.index = pd.DatetimeIndex(today_df.index).normalize()
    today_df.index.name = "date"
    return today_df


def _score_text(text: str, analyzer) -> float:
    """Return sentiment score in [-1, +1]. Uses VADER if available."""
    if analyzer is not None:
        scores = analyzer.polarity_scores(text)
        return float(scores["compound"])   # compound ∈ [-1, +1]

    # Simple keyword fallback
    text_lower = text.lower()
    bull_words  = ["bull", "buy", "moon", "pump", "rally", "up", "hodl", "long"]
    bear_words  = ["bear", "sell", "crash", "dump", "down", "rekt", "short"]
    bull_count  = sum(1 for w in bull_words if w in text_lower)
    bear_count  = sum(1 for w in bear_words if w in text_lower)
    total = bull_count + bear_count
    return (bull_count - bear_count) / total if total > 0 else 0.0


# ── Cache helpers ──────────────────────────────────────────────────────────────
# _load_cache, _save_cache, _filter_dates imported from data.cache_utils above
