"""
models/reddit_collector.py — Reddit data collection via PRAW.

Fetches posts + top-level comments from r/CryptoCurrency.
Caches daily to parquet files in ~/crypto-data/reddit/.

Setup:
    1. reddit.com/prefs/apps → "create another app" → type: script
    2. Set env vars (add to ~/.zshrc or .env):
         export REDDIT_CLIENT_ID=your_client_id
         export REDDIT_CLIENT_SECRET=your_client_secret
         export REDDIT_USER_AGENT="crypto-engine/1.0 by u/yourusername"
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import datetime, timezone, date

import pandas as pd

try:
    import praw
    _PRAW = True
except ImportError:
    _PRAW = False

DATA_CACHE  = Path(os.environ.get("CRYPTO_DATA_DIR", str(Path.home() / "crypto-data")))
REDDIT_DIR  = DATA_CACHE / "reddit"


# ── Auth ──────────────────────────────────────────────────────────────────────

def _get_reddit() -> "praw.Reddit":
    if not _PRAW:
        raise ImportError("praw not installed. Run: pip install praw")
    cid = os.environ.get("REDDIT_CLIENT_ID")
    sec = os.environ.get("REDDIT_CLIENT_SECRET")
    ua  = os.environ.get("REDDIT_USER_AGENT", "crypto-engine/1.0")
    if not cid or not sec:
        raise ValueError(
            "Reddit credentials missing.\n"
            "  export REDDIT_CLIENT_ID=<id>\n"
            "  export REDDIT_CLIENT_SECRET=<secret>"
        )
    return praw.Reddit(client_id=cid, client_secret=sec, user_agent=ua)


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_posts(
    subreddit: str = "CryptoCurrency",
    limit: int = 200,
    time_filter: str = "day",      # hour | day | week | month
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch top + hot posts from subreddit. Caches daily.

    Returns DataFrame: post_id, title, selftext, score, upvote_ratio,
                       num_comments, author, created_utc, flair
    """
    today_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache_path = REDDIT_DIR / f"posts_{today_str}.parquet"

    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    reddit = _get_reddit()
    sub    = reddit.subreddit(subreddit)
    rows: list[dict] = []
    seen: set[str]   = set()

    def _row(post) -> dict:
        return {
            "post_id":      post.id,
            "title":        post.title,
            "selftext":     (post.selftext or "")[:2000],
            "score":        post.score,
            "upvote_ratio": post.upvote_ratio,
            "num_comments": post.num_comments,
            "author":       str(post.author) if post.author else "[deleted]",
            "created_utc":  datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
            "flair":        post.link_flair_text or "",
        }

    # Top posts (highest score in time window)
    for post in sub.top(time_filter=time_filter, limit=limit):
        rows.append(_row(post))
        seen.add(post.id)

    # Hot posts (recency-weighted, catches fast-rising discussions)
    for post in sub.hot(limit=min(limit, 100)):
        if post.id not in seen:
            rows.append(_row(post))
            seen.add(post.id)

    # New posts (very recent, may not have score yet)
    for post in sub.new(limit=50):
        if post.id not in seen:
            rows.append(_row(post))
            seen.add(post.id)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        REDDIT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)

    return df


def fetch_comments(
    post_ids: list[str],
    comments_per_post: int = 25,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch top-level comments from a list of post IDs.

    Returns DataFrame: comment_id, post_id, body, score, author, created_utc
    """
    today_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache_path = REDDIT_DIR / f"comments_{today_str}.parquet"

    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        return cached[cached["post_id"].isin(set(post_ids))].copy()

    reddit = _get_reddit()
    rows: list[dict] = []

    for pid in post_ids:
        try:
            submission = reddit.submission(id=pid)
            submission.comments.replace_more(limit=0)   # no MoreComments expansion
            top_comments = sorted(
                submission.comments.list(),
                key=lambda c: getattr(c, "score", 0),
                reverse=True,
            )[:comments_per_post]

            for c in top_comments:
                body = getattr(c, "body", "")
                if not body or body in ("[deleted]", "[removed]"):
                    continue
                rows.append({
                    "comment_id": c.id,
                    "post_id":    pid,
                    "body":       body[:1000],
                    "score":      c.score,
                    "author":     str(c.author) if c.author else "[deleted]",
                    "created_utc": datetime.fromtimestamp(c.created_utc, tz=timezone.utc),
                })
            time.sleep(0.3)   # courtesy rate limiting
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
        REDDIT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)

    return df


# ── History ───────────────────────────────────────────────────────────────────

def load_post_history(days_back: int = 30) -> pd.DataFrame:
    """Load cached post data for the last N days."""
    REDDIT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp.utcnow() - pd.DateOffset(days=days_back)
    frames = []

    for f in sorted(REDDIT_DIR.glob("posts_*.parquet")):
        try:
            date_str = f.stem.replace("posts_", "")
            if pd.Timestamp(date_str) >= cutoff:
                frames.append(pd.read_parquet(f))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["post_id"])
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    return df.sort_values("created_utc").reset_index(drop=True)


def load_comment_history(days_back: int = 30) -> pd.DataFrame:
    """Load cached comment data for the last N days."""
    REDDIT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp.utcnow() - pd.DateOffset(days=days_back)
    frames = []

    for f in sorted(REDDIT_DIR.glob("comments_*.parquet")):
        try:
            date_str = f.stem.replace("comments_", "")
            if pd.Timestamp(date_str) >= cutoff:
                frames.append(pd.read_parquet(f))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["comment_id"])
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    return df.sort_values("created_utc").reset_index(drop=True)


# ── Volume & Attention Metrics ────────────────────────────────────────────────

def compute_volume_metrics(history_df: pd.DataFrame) -> dict:
    """
    Compute volume and attention metrics from historical post DataFrame.

    Returns:
      posts_today, posts_7d_avg, posts_change_pct,
      comment_velocity, unique_users, attention_label,
      top_post_title, top_post_score, volume_7d (list of daily counts)
    """
    if history_df.empty:
        return {"attention_label": "NO DATA"}

    df = history_df.copy()
    df["date"] = df["created_utc"].dt.date

    today     = datetime.now(timezone.utc).date()
    today_df  = df[df["date"] == today]
    past_df   = df[df["date"] != today]

    posts_today = len(today_df)

    by_day  = past_df.groupby("date").size()
    avg_7d  = float(by_day.tail(7).mean()) if len(by_day) >= 2 else posts_today
    chg_pct = (posts_today - avg_7d) / avg_7d * 100 if avg_7d > 0 else 0.0

    # Comment velocity: total comments across today's posts / hours elapsed
    hours_elapsed     = max(1, datetime.now(timezone.utc).hour + 1)
    total_comments    = int(today_df["num_comments"].sum()) if not today_df.empty else 0
    comment_velocity  = round(total_comments / hours_elapsed, 1)

    unique_users = today_df["author"].nunique() if not today_df.empty else 0

    # 7-day volume list (for sparkline)
    all_days   = sorted(by_day.index)[-7:]
    volume_7d  = [int(by_day.get(d, 0)) for d in all_days]

    # Top post today
    top_post_title = ""
    top_post_score = 0
    if not today_df.empty:
        top = today_df.sort_values("score", ascending=False).iloc[0]
        top_post_title = str(top["title"])[:80]
        top_post_score = int(top["score"])

    if chg_pct > 60:
        attention_label = "SPIKE"
    elif chg_pct > 25:
        attention_label = "ELEVATED"
    elif chg_pct < -35:
        attention_label = "LOW"
    else:
        attention_label = "NORMAL"

    return {
        "posts_today":      posts_today,
        "posts_7d_avg":     round(avg_7d, 1),
        "posts_change_pct": round(chg_pct, 1),
        "comment_velocity": comment_velocity,
        "unique_users":     unique_users,
        "attention_label":  attention_label,
        "top_post_title":   top_post_title,
        "top_post_score":   top_post_score,
        "volume_7d":        volume_7d,
    }
