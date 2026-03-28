"""
models/reddit_narrative.py — Narrative phase detection for Reddit crypto discourse.

Two-layer approach:

  Layer 1 — BERTopic (if installed)
            Discovers emergent semantic clusters week-over-week.
            Requires: pip install bertopic sentence-transformers umap-learn hdbscan
            First run downloads ~90 MB model to ~/.cache.

  Layer 2 — Keyword phase scorer (always available)
            Maps text to one of 5 crypto narrative phases based on weighted
            keyword matching. Fast, interpretable, no external models.

Narrative phases (in cycle order):
  1. CAPITULATION   — "crypto is dead", giving up
  2. SKEPTICISM     — "maybe undervalued", cautious watchers
  3. RECOVERY       — "accumulating", "buying the dip"
  4. OPTIMISM       — "institutions here", "real adoption"
  5. EUPHORIA       — "this time different", "moon", peak greed

Usage:
    from models.reddit_narrative import score_narrative_phase, detect_transition

    posts_df = ...  # with 'title', 'selftext', 'created_utc'
    weekly   = score_narrative_phase(posts_df)
    signal   = detect_transition(weekly)
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from bertopic import BERTopic
    _BERTOPIC = True
except ImportError:
    _BERTOPIC = False


# ── Phase definitions ─────────────────────────────────────────────────────────

PHASES = {
    "capitulation": {
        "order":       1,
        "label":       "CAPITULATION",
        "description": "Crypto is dead — giving up",
        "color":       "red",
        "keywords": [
            r"\bdead\b", r"\bover\b", r"\bfinished\b", r"never again",
            r"\brekt\b", r"lost everything", r"giving up", r"not coming back",
            r"waste of time", r"scam all along", r"told you so",
            r"bear market forever", r"going to zero", r"worthless",
            r"get out while you can", r"sold.{0,10}all",
            r"done with crypto", r"exit.{0,5}scam",
        ],
    },
    "skepticism": {
        "order":       2,
        "label":       "SKEPTICISM",
        "description": "Maybe this is undervalued — cautious watchers",
        "color":       "orange",
        "keywords": [
            r"maybe.{0,10}(undervalued|bottom)", r"could be.{0,10}bottom",
            r"not sure", r"risky", r"too early to tell",
            r"watching.{0,10}(from|closely|carefully)", r"waiting for confirmation",
            r"cautious", r"skeptic", r"interesting.{0,10}(level|price)",
            r"still risky", r"could go either way", r"monitoring",
            r"wait and see",
        ],
    },
    "recovery": {
        "order":       3,
        "label":       "RECOVERY",
        "description": "Buying the dip — accumulation mode",
        "color":       "yellow",
        "keywords": [
            r"\baccumulating\b", r"buying.{0,10}dip", r"adding.{0,5}position",
            r"\bdca\b", r"dollar.?cost", r"great opportunity",
            r"strong hands", r"fundamentals unchanged", r"fundamentals.{0,10}solid",
            r"long.?term.{0,5}perspective", r"bear market.{0,10}buy",
            r"stacking", r"loading up", r"buying.{0,5}more",
            r"nice.{0,5}entry", r"support.{0,5}holds",
        ],
    },
    "optimism": {
        "order":       4,
        "label":       "OPTIMISM",
        "description": "Institutions are here — fundamental adoption",
        "color":       "blue",
        "keywords": [
            r"institution", r"\betf\b", r"\badoption\b", r"mainstream",
            r"building", r"infrastructure", r"real.{0,5}world.{0,5}use",
            r"fundamentals.{0,10}strong", r"this is different because",
            r"maturity", r"legitimate.{0,5}asset", r"regulation.{0,5}(clear|good|positive)",
            r"corporate.{0,5}treasury", r"(banks?|fund).{0,10}buying",
            r"on.?chain.{0,5}(metric|data|activity)",
        ],
    },
    "euphoria": {
        "order":       5,
        "label":       "EUPHORIA",
        "description": "This time is different — peak greed signal",
        "color":       "purple",
        "keywords": [
            r"this time.{0,10}different", r"only.{0,5}going up",
            r"can'?t go down", r"\bmoon\b", r"supercycle",
            r"hyperbitcoin", r"\b100k\b", r"\b1.{0,3}million\b",
            r"early.{0,5}days", r"we'?re early",
            r"not financial advice but buy", r"wagmi",
            r"generational wealth", r"\blambo\b", r"wen lambo",
            r"all.?time.?high.{0,15}(soon|incoming|imminent)",
        ],
    },
}

_COMPILED: dict[str, list[re.Pattern]] = {
    phase: [re.compile(kw, re.IGNORECASE) for kw in data["keywords"]]
    for phase, data in PHASES.items()
}


# ── Phase Keyword Scorer ──────────────────────────────────────────────────────

def score_text_phases(text: str) -> dict[str, float]:
    """
    Return a score for each narrative phase for a single text.
    Higher score = stronger presence of that phase's language.
    """
    t      = (text or "").lower()
    scores = {}
    for phase, patterns in _COMPILED.items():
        scores[phase] = sum(1.0 for p in patterns if p.search(t))
    return scores


def score_posts_phases(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-phase score columns to posts DataFrame.

    Input:  DataFrame with 'title' and 'selftext' columns
    Output: Same + phase_capitulation, phase_skepticism, phase_recovery,
                   phase_optimism, phase_euphoria, dominant_phase
    """
    if posts_df.empty:
        return posts_df

    df    = posts_df.copy()
    texts = (df["title"].fillna("") + " " + df["selftext"].fillna("")).tolist()

    all_scores = [score_text_phases(t) for t in texts]

    for phase in PHASES:
        df[f"phase_{phase}"] = [s[phase] for s in all_scores]

    phase_cols  = [f"phase_{p}" for p in PHASES]
    totals      = df[phase_cols].sum(axis=1)
    df["dominant_phase"] = df[phase_cols].idxmax(axis=1).str.replace("phase_", "")
    df["dominant_phase"] = df["dominant_phase"].where(totals > 0, "neutral")

    return df


# ── Weekly Aggregation ────────────────────────────────────────────────────────

def score_narrative_phase(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate narrative phase scores to weekly buckets.

    Returns DataFrame indexed by week_start (Monday) with columns:
      n_posts, phase_capitulation ... phase_euphoria (raw counts),
      pct_capitulation ... pct_euphoria (% of scored posts),
      dominant_phase, phase_score (1-5 weighted average),
      phase_entropy (diversity measure)
    """
    if posts_df.empty:
        return pd.DataFrame()

    df = score_posts_phases(posts_df)
    df["week"] = pd.to_datetime(df["created_utc"]).dt.to_period("W").apply(lambda p: p.start_time)

    def _agg_week(g):
        n = len(g)
        phase_counts = {p: float(g[f"phase_{p}"].sum()) for p in PHASES}
        total_mentions = sum(phase_counts.values()) or 1.0

        pcts   = {p: phase_counts[p] / total_mentions * 100 for p in PHASES}
        orders = {p: PHASES[p]["order"] for p in PHASES}

        # Weighted average phase score (1–5)
        phase_score = sum(pcts[p] / 100 * orders[p] for p in PHASES)

        # Entropy (high = mixed signals, low = clear dominant narrative)
        vals = np.array([pcts[p] / 100 for p in PHASES])
        vals = vals[vals > 0]
        entropy = float(-np.sum(vals * np.log(vals))) if len(vals) > 0 else 0.0

        dominant = max(pcts, key=lambda p: pcts[p])

        row = {"n_posts": n, "phase_score": round(phase_score, 2), "phase_entropy": round(entropy, 3),
               "dominant_phase": dominant}
        for p in PHASES:
            row[f"pct_{p}"] = round(pcts[p], 1)
        return pd.Series(row)

    weekly = df.groupby("week").apply(_agg_week)
    return weekly.sort_index()


# ── Transition Detection ──────────────────────────────────────────────────────

_TRANSITIONS = [
    ("capitulation", "skepticism",  "Bottom signal: fear giving way to watching"),
    ("skepticism",   "recovery",    "Early bull: cautious watching → active accumulation"),
    ("recovery",     "optimism",    "Mid bull: accumulation → institutional narrative"),
    ("optimism",     "euphoria",    "Late bull: fundamentals → 'this time different' 🚨"),
    ("euphoria",     "optimism",    "Post-peak: euphoria cooling, still optimistic"),
    ("optimism",     "skepticism",  "Topping: optimism fading, skepticism creeping in"),
    ("skepticism",   "capitulation","Bear deepening: skeptics becoming sellers"),
]


def detect_transition(weekly_df: pd.DataFrame, window: int = 3) -> dict:
    """
    Detect narrative phase transitions in the last N weeks.

    Returns:
      current_phase, current_phase_score, prev_phase, transitioning (bool),
      transition_description, confidence, history (list of (week, phase, score))
    """
    if weekly_df.empty or len(weekly_df) < 2:
        return {"current_phase": "unknown", "transitioning": False}

    recent = weekly_df.tail(window)

    current_phase = str(recent["dominant_phase"].iloc[-1])
    prev_phase    = str(recent["dominant_phase"].iloc[-2]) if len(recent) >= 2 else current_phase
    current_score = float(recent["phase_score"].iloc[-1])

    # Detect transition: dominant phase changed or phase_score moved significantly
    phase_changed    = current_phase != prev_phase
    score_delta      = float(recent["phase_score"].diff().iloc[-1]) if len(recent) >= 2 else 0.0
    score_moving     = abs(score_delta) > 0.3

    transitioning = phase_changed or score_moving

    transition_description = ""
    for from_p, to_p, desc in _TRANSITIONS:
        if (phase_changed and prev_phase == from_p and current_phase == to_p):
            transition_description = desc
            break
    if not transition_description and score_moving:
        direction = "rising" if score_delta > 0 else "falling"
        transition_description = f"Phase score {direction} ({score_delta:+.2f}/week)"

    # Confidence: how dominant is the leading phase in the current week?
    pct_current = float(recent[f"pct_{current_phase}"].iloc[-1]) if f"pct_{current_phase}" in recent.columns else 0.0
    confidence  = round(min(pct_current / 50 * 100, 100), 0)

    history = [
        (str(idx.date()), row["dominant_phase"], round(row["phase_score"], 2))
        for idx, row in weekly_df.tail(8).iterrows()
    ]

    return {
        "current_phase":         current_phase,
        "current_phase_label":   PHASES.get(current_phase, {}).get("label", current_phase.upper()),
        "current_phase_desc":    PHASES.get(current_phase, {}).get("description", ""),
        "current_phase_score":   current_score,
        "prev_phase":            prev_phase,
        "transitioning":         transitioning,
        "transition_description": transition_description,
        "score_delta":           round(score_delta, 3),
        "confidence":            confidence,
        "history":               history,
    }


# ── BERTopic (optional deep layer) ───────────────────────────────────────────

def fit_bertopic(texts: Sequence[str], nr_topics: int = 10) -> tuple | None:
    """
    Fit BERTopic to discover semantic topic clusters.

    Returns (topic_model, topic_info_df, topics_per_doc) or None if unavailable.
    Requires: pip install bertopic sentence-transformers umap-learn hdbscan
    """
    if not _BERTOPIC:
        return None
    if len(texts) < 20:
        return None

    try:
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        umap_model      = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                               metric="cosine", random_state=42)
        hdbscan_model   = HDBSCAN(min_cluster_size=max(5, len(texts) // 20),
                                  metric="euclidean",
                                  cluster_selection_method="eom",
                                  prediction_data=True)

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=nr_topics,
            verbose=False,
        )

        topics, _  = topic_model.fit_transform(list(texts))
        topic_info = topic_model.get_topic_info()

        return topic_model, topic_info, topics

    except Exception as e:
        return None


def bertopic_weekly_themes(posts_df: pd.DataFrame, nr_topics: int = 8) -> list[dict] | None:
    """
    Use BERTopic to extract top themes from posts in the last 7 days.

    Returns list of {topic_id, label, keywords, count, pct} or None.
    """
    if posts_df.empty:
        return None

    df    = posts_df.copy()
    texts = (df["title"].fillna("") + " " + df["selftext"].fillna("")).tolist()
    texts = [t for t in texts if len(t.strip()) > 20]

    result = fit_bertopic(texts, nr_topics=nr_topics)
    if result is None:
        return None

    topic_model, topic_info, topics = result
    total = len(texts)

    themes = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:   # outlier cluster
            continue
        kws   = topic_model.get_topic(tid)
        label = " / ".join(w for w, _ in kws[:3]) if kws else f"Topic {tid}"
        count = int(row["Count"])
        themes.append({
            "topic_id": tid,
            "label":    label,
            "keywords": [w for w, _ in kws[:6]] if kws else [],
            "count":    count,
            "pct":      round(count / total * 100, 1),
        })

    return sorted(themes, key=lambda t: t["count"], reverse=True)[:6]


# ── Keyword-based top narratives (fallback for BERTopic) ────────────────────

_NARRATIVE_BUCKETS = {
    "macro headwinds":         [r"macro", r"fed", r"interest rate", r"inflation", r"recession", r"economy"],
    "accumulation opportunity":[r"accumul", r"buying.{0,5}dip", r"\bdca\b", r"loading up", r"stacking"],
    "regulatory uncertainty":  [r"regulation", r"sec", r"crackdown", r"ban", r"legal"],
    "institutional activity":  [r"institution", r"corporate", r"fund", r"\betf\b", r"blackrock", r"fidelity"],
    "technical analysis":      [r"support", r"resistance", r"chart", r"\bta\b", r"fibonacci", r"rsi", r"macd"],
    "on-chain strength":       [r"on.?chain", r"wallet", r"addresses", r"hash.?rate", r"miner", r"network"],
    "altcoin rotation":        [r"altcoin", r"alt.?season", r"\beth\b", r"sol", r"rotation"],
    "fear & uncertainty":      [r"\bfud\b", r"crash", r"dump", r"panic", r"liquidat"],
    "halving narrative":       [r"halving", r"halvening", r"supply shock", r"miner reward"],
    "stablecoin / cash":       [r"stablecoin", r"usdc", r"usdt", r"cash.{0,5}(is|was) king", r"sitting in cash"],
}

_NARRATIVE_COMPILED = {
    name: [re.compile(p, re.IGNORECASE) for p in patterns]
    for name, patterns in _NARRATIVE_BUCKETS.items()
}

_NARRATIVE_SENTIMENT = {
    "macro headwinds":          "bearish",
    "accumulation opportunity": "bullish",
    "regulatory uncertainty":   "bearish",
    "institutional activity":   "bullish",
    "technical analysis":       "neutral",
    "on-chain strength":        "bullish",
    "altcoin rotation":         "neutral",
    "fear & uncertainty":       "bearish",
    "halving narrative":        "bullish",
    "stablecoin / cash":        "bearish",
}


def keyword_top_narratives(posts_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Identify top narratives from posts using keyword bucket matching.
    Always available (no external models required).

    Returns list of {name, count, pct, sentiment} sorted by count.
    """
    if posts_df.empty:
        return []

    texts  = (posts_df["title"].fillna("") + " " + posts_df["selftext"].fillna("")).tolist()
    counts: Counter = Counter()

    for text in texts:
        t = text.lower()
        for name, patterns in _NARRATIVE_COMPILED.items():
            if any(p.search(t) for p in patterns):
                counts[name] += 1

    total = len(texts) or 1
    results = [
        {
            "name":      name,
            "count":     count,
            "pct":       round(count / total * 100, 1),
            "sentiment": _NARRATIVE_SENTIMENT.get(name, "neutral"),
        }
        for name, count in counts.most_common(top_n)
        if count > 0
    ]
    return results
