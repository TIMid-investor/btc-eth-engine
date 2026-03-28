#!/usr/bin/env python3
"""
scripts/run_reddit.py — r/CryptoCurrency Sentiment Dashboard

Prints a three-panel dashboard:
  A. Volume & Attention  — post/comment spikes, unique users
  B. Sentiment Polarity  — bullish/bearish %, emotion breakdown
  C. Narrative Phase     — cycle phase detection + top narratives

Usage:
    # Fetch fresh data and show dashboard
    python scripts/run_reddit.py --fetch

    # Use cached data (default, no Reddit API call)
    python scripts/run_reddit.py

    # Use keyword scorer only (no transformers required)
    python scripts/run_reddit.py --no-finbert

    # Historical window
    python scripts/run_reddit.py --days 14

    # Fetch comments for deeper sentiment (slower)
    python scripts/run_reddit.py --fetch --comments

    # BERTopic for richer topic discovery (if installed)
    python scripts/run_reddit.py --bertopic
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.reddit_collector import (
    fetch_posts, fetch_comments,
    load_post_history, load_comment_history,
    compute_volume_metrics,
)
from models.reddit_sentiment import (
    score_posts_df, aggregate_daily,
    sentiment_label_from_compound,
)
from models.reddit_narrative import (
    score_narrative_phase, detect_transition,
    keyword_top_narratives, bertopic_weekly_themes,
    PHASES,
)


# ── Display helpers ───────────────────────────────────────────────────────────

W = 66   # panel width

def _bar(pct: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    filled = round(pct / 100 * width)
    return fill * filled + empty * (width - filled)


def _sparkline(values: list[float], width: int = 14) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng    = mx - mn or 1
    return "".join(blocks[min(8, int((v - mn) / rng * 8))] for v in values[-width:])


def _header(title: str) -> None:
    print()
    print("═" * W)
    pad = (W - len(title) - 2) // 2
    print(" " * pad + f" {title} ")
    print("═" * W)


def _panel(title: str) -> None:
    inner = W - 4
    print(f"┌─ {title} {'─' * max(0, inner - len(title) - 1)}┐")


def _panel_end() -> None:
    print("└" + "─" * (W - 2) + "┘")


def _row(text: str = "") -> None:
    print(f"│  {text:<{W - 5}}│")


# ── Panel A — Volume & Attention ─────────────────────────────────────────────

def print_volume_panel(posts_df: pd.DataFrame) -> None:
    vm = compute_volume_metrics(posts_df)

    _panel("A. VOLUME & ATTENTION")

    label     = vm.get("attention_label", "UNKNOWN")
    chg       = vm.get("posts_change_pct", 0.0)
    arrow     = "▲" if chg >= 0 else "▼"
    chg_str   = f"{chg:+.0f}%"
    attn_str  = f"{label} {arrow}"

    _row(f"Posts today:        {vm.get('posts_today', 0):<6}  (7d avg: {vm.get('posts_7d_avg', 0):.0f})  {chg_str}  {attn_str}")
    _row(f"Comments / hour:    {vm.get('comment_velocity', 0):,.0f}")
    _row(f"Unique authors:     {vm.get('unique_users', 0):,}")
    _row()

    vol7 = vm.get("volume_7d", [])
    if vol7:
        spark = _sparkline(vol7, width=14)
        _row(f"7-day post volume:  {spark}  trend: {'RISING' if chg > 10 else 'FALLING' if chg < -10 else 'FLAT'}")
        _row()

    top_title = vm.get("top_post_title", "")
    top_score = vm.get("top_post_score", 0)
    if top_title:
        _row(f"Top post (score {top_score:,}):")
        # Word-wrap at W-8
        line_w = W - 8
        words = top_title.split()
        line  = ""
        for word in words:
            if len(line) + len(word) + 1 <= line_w:
                line = (line + " " + word).strip()
            else:
                _row(f"  {line}")
                line = word
        if line:
            _row(f"  {line}")

    _panel_end()


# ── Panel B — Sentiment Polarity ─────────────────────────────────────────────

def print_sentiment_panel(scored_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    _panel("B. SENTIMENT POLARITY")

    if scored_df.empty:
        _row("No scored data available.")
        _panel_end()
        return

    # Today / recent aggregate
    today_data = scored_df[
        scored_df["created_utc"].dt.date == datetime.now(timezone.utc).date()
    ] if not scored_df.empty else scored_df

    if today_data.empty:
        today_data = scored_df  # fall back to all available

    compound  = float(today_data["compound"].mean())
    p_bull    = (today_data["sentiment_label"] == "bullish").mean() * 100
    p_bear    = (today_data["sentiment_label"] == "bearish").mean() * 100
    p_neu     = (today_data["sentiment_label"] == "neutral").mean() * 100
    label_str = sentiment_label_from_compound(compound)
    model_str = today_data["score_model"].iloc[0] if not today_data.empty else "unknown"

    _row(f"Overall:  {label_str:<30} ({compound:+.3f})")
    _row(f"Model:    {model_str}")
    _row()
    _row(f"Bullish    {_bar(p_bull, 18)}  {p_bull:.0f}%")
    _row(f"Neutral    {_bar(p_neu,  18)}  {p_neu:.0f}%")
    _row(f"Bearish    {_bar(p_bear, 18)}  {p_bear:.0f}%")
    _row()

    # Emotion breakdown
    if "emotion" in today_data.columns:
        from collections import Counter
        emotion_counts = Counter(today_data["emotion"])
        total_emotions = len(today_data)
        _row("Emotion breakdown:")
        emotion_order = ["capitulation", "fud", "cautious_optimism", "fomo", "euphoria", "neutral"]
        emotion_labels = {
            "capitulation":    "Capitulation",
            "fud":             "FUD / Fear",
            "cautious_optimism":"Cautious opt.",
            "fomo":            "FOMO",
            "euphoria":        "Euphoria",
            "neutral":         "Neutral",
        }
        emotion_notes = {
            "capitulation": " ← elevated = bottom forming",
            "euphoria":     " ← 🚨 peak greed signal",
            "fud":          "",
            "fomo":         " ← late-cycle if dominant",
            "cautious_optimism": " ← healthy bull signal",
            "neutral":      "",
        }
        for emo in emotion_order:
            cnt  = emotion_counts.get(emo, 0)
            pct  = cnt / total_emotions * 100 if total_emotions > 0 else 0
            note = emotion_notes.get(emo, "")
            if pct > 0:
                _row(f"  {emotion_labels[emo]:<18} {_bar(pct, 12)}  {pct:.0f}%{note}")
    _row()

    # Historical trend
    if not daily_df.empty and len(daily_df) >= 7:
        avg30 = float(daily_df["weighted_compound"].tail(30).mean()) if len(daily_df) >= 30 else None
        avg7  = float(daily_df["weighted_compound"].tail(7).mean())
        spark = _sparkline(daily_df["weighted_compound"].tail(14).tolist(), width=14)
        trend_dir = "IMPROVING ▲" if compound > avg7 else "DECLINING ▼" if compound < avg7 else "STABLE"
        _row(f"14-day trend:  {spark}  {trend_dir}")
        if avg30 is not None:
            _row(f"Today: {compound:+.3f}  |  7d avg: {avg7:+.3f}  |  30d avg: {avg30:+.3f}")

    _panel_end()


# ── Panel C — Narrative Phase ─────────────────────────────────────────────────

def print_narrative_panel(
    posts_df: pd.DataFrame,
    use_bertopic: bool = False,
) -> None:
    _panel("C. NARRATIVE PHASE DETECTION")

    if posts_df.empty:
        _row("No post data available.")
        _panel_end()
        return

    weekly = score_narrative_phase(posts_df)
    signal = detect_transition(weekly)

    current = signal.get("current_phase", "unknown")
    c_label = signal.get("current_phase_label", current.upper())
    c_desc  = signal.get("current_phase_desc", "")
    c_score = signal.get("current_phase_score", 0.0)
    trans   = signal.get("transitioning", False)
    t_desc  = signal.get("transition_description", "")
    conf    = signal.get("confidence", 0)

    # Phase score bar (1-5)
    phase_bar_filled = round((c_score - 1) / 4 * 20)
    phase_bar = "█" * max(0, phase_bar_filled) + "░" * max(0, 20 - phase_bar_filled)

    status = f"  ← {t_desc}" if trans else ""
    _row(f"Current phase:  {c_label}")
    _row(f"  \"{c_desc}\"")
    _row(f"Phase score:    {c_score:.1f} / 5.0    [{phase_bar}]{status}")
    _row(f"Confidence:     {conf:.0f}%")
    _row()

    # Weekly phase history
    history = signal.get("history", [])
    if history:
        _row("Weekly phase history (oldest → newest):")
        max_score = max(s for _, _, s in history) if history else 5
        for week_str, phase, score in history[-6:]:
            phase_label = PHASES.get(phase, {}).get("label", phase.upper())[:14]
            bar_w       = round(score / 5.0 * 20)
            bar         = "█" * bar_w + "░" * (20 - bar_w)
            marker      = " ◄ NOW" if (week_str, phase, score) == history[-1] else ""
            _row(f"  {week_str}  {phase_label:<14}  {bar}{marker}")
    _row()

    # Top narratives
    _row("Top narratives this week:")

    narratives = None
    if use_bertopic:
        today_posts = posts_df[
            posts_df["created_utc"].dt.date >= (
                pd.Timestamp.utcnow() - pd.DateOffset(days=7)
            ).date()
        ]
        narratives = bertopic_weekly_themes(today_posts, nr_topics=8)
        if narratives:
            for i, t in enumerate(narratives[:5], 1):
                _row(f"  {i}. \"{t['label']}\" ({t['pct']:.0f}%)")

    if not narratives:
        week_posts = posts_df[
            posts_df["created_utc"].dt.date >= (
                pd.Timestamp.utcnow() - pd.DateOffset(days=7)
            ).date()
        ]
        kwn = keyword_top_narratives(week_posts, top_n=5)
        if kwn:
            sent_arrow = {"bullish": "↑", "bearish": "↓", "neutral": "→"}
            for i, n in enumerate(kwn, 1):
                arrow = sent_arrow.get(n["sentiment"], "")
                _row(f"  {i}. \"{n['name']}\" ({n['pct']:.0f}%)  {arrow} {n['sentiment']}")
        else:
            _row("  (insufficient data for narrative detection)")
    _row()

    # Key signal / alert
    if trans and t_desc:
        alert_symbol = "⚠" if c_score >= 4.0 else "→"
        _row(f"{alert_symbol} TRANSITION SIGNAL: {t_desc}")

    # Phase-specific guidance
    guidance = {
        "capitulation":   "Historically a bottom-forming zone. High fear = opportunity.",
        "skepticism":     "Smart money accumulation phase. Watch for recovery confirmation.",
        "recovery":       "Early bull signs. DCA into strength, hold through volatility.",
        "optimism":       "Mid-bull. Good gains ahead but start thinking about exits.",
        "euphoria":       "🚨 LATE CYCLE. Consider scaling out. 'This time different' = danger.",
    }
    if current in guidance:
        _row()
        _row(f"Guidance: {guidance[current]}")

    _panel_end()


# ── Sentiment trend chart (ASCII) ─────────────────────────────────────────────

def print_trend_chart(daily_df: pd.DataFrame, days: int = 30) -> None:
    """Print a simple ASCII chart of daily sentiment over time."""
    if daily_df.empty or len(daily_df) < 3:
        return

    data  = daily_df["weighted_compound"].tail(days).dropna()
    dates = [str(d.date()) for d in data.index]
    vals  = data.tolist()

    if not vals:
        return

    print()
    print(f"  Sentiment trend — last {len(vals)} days")
    print(f"  {'─' * 50}")

    # Normalise to rows
    mn, mx = min(vals), max(vals)
    rng    = mx - mn or 0.001
    height = 6

    rows = []
    for row_i in range(height, -1, -1):
        threshold = mn + (row_i / height) * rng
        line = ""
        for v in vals[-40:]:  # cap at 40 chars wide
            if v >= threshold:
                line += "▄"
            else:
                line += " "
        label = f"{threshold:+.2f}" if row_i in (height, height // 2, 0) else "     "
        rows.append(f"  {label} │{line}│")

    for r in rows:
        print(r)
    print(f"         └{'─' * min(40, len(vals))}┘")
    print(f"           {dates[0][:10]}{'':>20}{dates[-1][:10]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="r/CryptoCurrency Sentiment Dashboard")
    parser.add_argument("--fetch",      action="store_true", help="Fetch fresh data from Reddit API")
    parser.add_argument("--comments",   action="store_true", help="Also fetch comments (slower)")
    parser.add_argument("--days",       type=int, default=30, help="Days of history to analyse (default: 30)")
    parser.add_argument("--no-finbert", action="store_true", help="Skip FinBERT, use VADER/keywords only")
    parser.add_argument("--bertopic",   action="store_true", help="Use BERTopic for narrative discovery")
    parser.add_argument("--subreddit",  default="CryptoCurrency", help="Subreddit to analyse")
    parser.add_argument("--limit",      type=int, default=200, help="Number of posts to fetch (default: 200)")
    args = parser.parse_args()

    prefer_finbert = not args.no_finbert

    _header(f"r/{args.subreddit} SENTIMENT DASHBOARD  —  {datetime.now(timezone.utc).strftime('%d %B %Y  %H:%M UTC')}")

    # ── Fetch / Load ──────────────────────────────────────────────────────────
    if args.fetch:
        print(f"\n  Fetching posts from r/{args.subreddit}...")
        try:
            today_posts = fetch_posts(
                subreddit=args.subreddit,
                limit=args.limit,
                time_filter="day",
                force_refresh=True,
            )
            print(f"  ✓ Fetched {len(today_posts)} posts")

            if args.comments and not today_posts.empty:
                top_ids = today_posts.head(30)["post_id"].tolist()
                print(f"  Fetching comments for top {len(top_ids)} posts...")
                fetch_comments(top_ids, force_refresh=True)
                print(f"  ✓ Comments cached")
        except Exception as e:
            print(f"  ✗ Fetch failed: {e}")
            print("  Falling back to cached data.\n")

    # Load history
    posts_df = load_post_history(days_back=args.days)

    if posts_df.empty:
        print("\n  No data available. Run with --fetch to download posts.\n")
        print(f"  Requirements: pip install praw")
        print(f"  Credentials:  export REDDIT_CLIENT_ID=<id>")
        print(f"                export REDDIT_CLIENT_SECRET=<secret>")
        return

    print(f"\n  Loaded {len(posts_df)} posts over {args.days} days")

    # ── Score sentiment ───────────────────────────────────────────────────────
    print(f"  Scoring sentiment ({'FinBERT' if prefer_finbert else 'VADER/keywords'})...", end="", flush=True)
    try:
        scored_df = score_posts_df(posts_df, prefer_finbert=prefer_finbert)
        daily_df  = aggregate_daily(scored_df)
        print(" done")
    except Exception as e:
        print(f" failed ({e}), using keyword scorer")
        scored_df = score_posts_df(posts_df, prefer_finbert=False)
        daily_df  = aggregate_daily(scored_df)

    # ── Print panels ──────────────────────────────────────────────────────────
    print()
    print_volume_panel(posts_df)
    print()
    print_sentiment_panel(scored_df, daily_df)
    print()
    print_narrative_panel(posts_df, use_bertopic=args.bertopic)

    # ── Sentiment trend ASCII chart ───────────────────────────────────────────
    if not daily_df.empty and len(daily_df) >= 5:
        print_trend_chart(daily_df, days=args.days)

    # ── Footer ────────────────────────────────────────────────────────────────
    print()
    print("─" * W)
    model_used = "FinBERT" if (prefer_finbert and not scored_df.empty
                               and scored_df["score_model"].iloc[0] == "finbert") else "VADER/keywords"
    print(f"  Sentiment model: {model_used}  |  Posts analysed: {len(posts_df)}")
    print(f"  Refresh:  python scripts/run_reddit.py --fetch")
    if args.bertopic:
        print(f"  Topics:   BERTopic (deep mode)")
    print("─" * W)
    print()


if __name__ == "__main__":
    main()
