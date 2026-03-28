"""
models/reddit_sentiment.py — NLP sentiment scoring for Reddit text.

Three-layer stack (best available is used automatically):

  Layer 1 — FinBERT (ProsusAI/finbert)
            Finance-domain BERT. Positive / Negative / Neutral probabilities.
            Requires: pip install transformers torch
            First run downloads ~400 MB model (cached to ~/.cache/huggingface).

  Layer 2 — VADER  (vaderSentiment)
            Rule-based, tuned for social media slang. Fast.
            Requires: pip install vaderSentiment

  Layer 3 — Keyword fallback
            Zero dependencies. Uses crypto-specific lexicon.

Emotion classification (beyond direction):
  EUPHORIA · FOMO · FUD · CAPITULATION · CAUTIOUS_OPTIMISM · NEUTRAL

Usage:
    from models.reddit_sentiment import score_texts, classify_emotion, aggregate_daily

    scores = score_texts(["BTC is dead", "buying the dip"])
    # [{"compound": -0.72, "label": "bearish", "confidence": 0.81}, ...]
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderAnalyzer
    _vader_instance = _VaderAnalyzer()
    # Augment VADER lexicon with crypto slang
    _crypto_lexicon = {
        "moon": 2.5, "mooning": 2.5, "rekt": -2.5, "rug": -2.8,
        "wagmi": 2.0, "ngmi": -2.0, "hodl": 1.5, "hodling": 1.5,
        "fud": -1.8, "fomo": 1.0, "shill": -1.2, "shilling": -1.2,
        "bagholder": -1.5, "lambo": 2.0, "btfd": 1.8,
        "capitulation": -2.0, "capitulated": -2.0,
        "accumulating": 1.5, "dca": 1.2, "averaging": 0.8,
    }
    _vader_instance.lexicon.update(_crypto_lexicon)
    _VADER = True
except ImportError:
    _VADER = False

try:
    from transformers import pipeline as _hf_pipeline, logging as _hf_logging
    import torch
    _hf_logging.set_verbosity_error()
    _FINBERT = True
    _finbert_pipe = None   # lazy-loaded on first use
except ImportError:
    _FINBERT = False
    _finbert_pipe = None


# ── Keyword fallback lexicon ──────────────────────────────────────────────────

_BULLISH_KEYWORDS = [
    "buy", "buying", "bull", "bullish", "moon", "pump", "ath", "green",
    "recovery", "rebound", "bounce", "accumulate", "accumulating", "dca",
    "hodl", "hold", "long", "undervalued", "cheap", "opportunity",
    "bottom", "bottomed", "support", "breakout", "all time high", "wagmi",
    "uptrend", "reversal", "institutional", "etf", "adoption",
]

_BEARISH_KEYWORDS = [
    "sell", "selling", "bear", "bearish", "crash", "dump", "drop", "red",
    "rekt", "rug", "scam", "ponzi", "dead", "over", "worthless", "bubble",
    "manipulation", "capitulation", "panic", "loss", "losing", "fud",
    "short", "overvalued", "breakdown", "support broke", "ngmi",
    "correction", "collapse", "liquidation", "cascading",
]


def _keyword_score(text: str) -> float:
    """Simple keyword polarity: +1/-1 per word, normalised to [-1, 1]."""
    t    = text.lower()
    bull = sum(1 for w in _BULLISH_KEYWORDS if w in t)
    bear = sum(1 for w in _BEARISH_KEYWORDS if w in t)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


# ── Emotion lexicon ───────────────────────────────────────────────────────────

_EMOTION_PATTERNS: dict[str, list[str]] = {
    "euphoria": [
        r"\bmoon\b", r"lambo", r"wagmi", r"this time.{0,15}different",
        r"100x", r"1 ?million", r"generational.{0,10}wealth",
        r"only.{0,5}up", r"supercycle", r"can'?t stop",
        r"number.{0,5}go.{0,5}up", r"wen.{0,5}moon", r"hyperbitcoin",
        r"we'?re all gonna make it",
    ],
    "fomo": [
        r"fomo", r"fear of missing", r"left behind", r"last chance",
        r"before.{0,10}pump", r"wish i (bought|had)", r"should.{0,10}bought",
        r"regret.{0,15}not buying", r"missed.{0,10}move",
    ],
    "fud": [
        r"\bscam\b", r"ponzi", r"going to zero", r"worthless",
        r"manipulation", r"rug.?pull", r"exit.?scam", r"whales.{0,10}dump",
        r"dead.?cat", r"sucker.?rally", r"bear.?trap",
        r"wash.?trading", r"fake.?volume",
    ],
    "capitulation": [
        r"sold.{0,10}everything", r"never again", r"done with crypto",
        r"\brekt\b", r"lost everything", r"taking.{0,5}losses",
        r"i give up", r"not worth it", r"crypto is dead",
        r"getting out", r"sold at a loss", r"lesson learned",
        r"all.?time low", r"can'?t take.{0,10}anymore",
    ],
    "cautious_optimism": [
        r"slowly accumul", r"\bdca\b", r"buying.{0,10}small",
        r"cautious", r"long.?term.{0,10}hold", r"patient",
        r"fundamentals.{0,10}(good|solid|strong|unchanged)",
        r"undervalued at.{0,10}(these|current).{0,5}price",
        r"good.{0,5}entry", r"reasonable.{0,5}(price|level)",
        r"dollar.?cost", r"averaging.{0,5}(in|down)",
    ],
}

_COMPILED_EMOTIONS: dict[str, list[re.Pattern]] = {
    emotion: [re.compile(p, re.IGNORECASE) for p in patterns]
    for emotion, patterns in _EMOTION_PATTERNS.items()
}


def classify_emotion(text: str) -> str:
    """
    Return dominant emotion label for a single text string.

    Returns one of: euphoria | fomo | fud | capitulation |
                    cautious_optimism | neutral
    """
    t      = text.lower()
    scores = {}
    for emotion, patterns in _COMPILED_EMOTIONS.items():
        scores[emotion] = sum(1 for p in patterns if p.search(t))

    if not any(scores.values()):
        return "neutral"
    return max(scores, key=lambda e: scores[e])


def classify_emotions_batch(texts: Sequence[str]) -> list[str]:
    return [classify_emotion(t) for t in texts]


# ── Sentiment scoring ─────────────────────────────────────────────────────────

def _load_finbert():
    global _finbert_pipe
    if _finbert_pipe is None:
        device = 0 if (torch.backends.mps.is_available() or torch.cuda.is_available()) else -1
        _finbert_pipe = _hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=3,
            device=device,
            truncation=True,
            max_length=512,
        )
    return _finbert_pipe


def _finbert_score(text: str) -> dict:
    pipe   = _load_finbert()
    result = pipe(text[:512])[0]
    probs  = {r["label"].lower(): r["score"] for r in result}
    pos    = probs.get("positive", 0.0)
    neg    = probs.get("negative", 0.0)
    neu    = probs.get("neutral",  0.0)
    compound = pos - neg
    if compound > 0.15:
        label = "bullish"
    elif compound < -0.15:
        label = "bearish"
    else:
        label = "neutral"
    confidence = max(pos, neg, neu)
    return {"compound": compound, "label": label, "confidence": confidence,
            "p_positive": pos, "p_negative": neg, "p_neutral": neu,
            "model": "finbert"}


def _vader_score(text: str) -> dict:
    vs       = _vader_instance.polarity_scores(text)
    compound = vs["compound"]
    if compound > 0.10:
        label = "bullish"
    elif compound < -0.10:
        label = "bearish"
    else:
        label = "neutral"
    confidence = abs(compound)
    return {"compound": compound, "label": label, "confidence": confidence,
            "p_positive": vs["pos"], "p_negative": vs["neg"], "p_neutral": vs["neu"],
            "model": "vader"}


def _keyword_score_dict(text: str) -> dict:
    compound   = _keyword_score(text)
    if compound > 0.1:
        label = "bullish"
    elif compound < -0.1:
        label = "bearish"
    else:
        label = "neutral"
    return {"compound": compound, "label": label, "confidence": abs(compound),
            "p_positive": max(0, compound), "p_negative": max(0, -compound),
            "p_neutral": 1.0 - abs(compound), "model": "keyword"}


def score_text(text: str, prefer_finbert: bool = True) -> dict:
    """
    Score a single text string.

    Returns dict: compound [-1,1], label (bullish/neutral/bearish),
                  confidence [0,1], p_positive, p_negative, p_neutral, model
    """
    if not text or not text.strip():
        return {"compound": 0.0, "label": "neutral", "confidence": 0.0,
                "p_positive": 0.0, "p_negative": 0.0, "p_neutral": 1.0,
                "model": "empty"}
    if prefer_finbert and _FINBERT:
        try:
            return _finbert_score(text)
        except Exception:
            pass
    if _VADER:
        return _vader_score(text)
    return _keyword_score_dict(text)


def score_texts(
    texts: Sequence[str],
    prefer_finbert: bool = True,
    batch_size: int = 32,
) -> list[dict]:
    """
    Score a list of texts. Uses batching for FinBERT efficiency.
    Falls back to VADER or keyword scorer.
    """
    if not texts:
        return []

    if prefer_finbert and _FINBERT:
        try:
            pipe    = _load_finbert()
            results = []
            for i in range(0, len(texts), batch_size):
                batch     = [t[:512] for t in texts[i : i + batch_size]]
                raw_batch = pipe(batch)
                for raw in raw_batch:
                    probs     = {r["label"].lower(): r["score"] for r in raw}
                    pos       = probs.get("positive", 0.0)
                    neg       = probs.get("negative", 0.0)
                    neu       = probs.get("neutral",  0.0)
                    compound  = pos - neg
                    label     = "bullish" if compound > 0.15 else "bearish" if compound < -0.15 else "neutral"
                    results.append({
                        "compound": compound, "label": label,
                        "confidence": max(pos, neg, neu),
                        "p_positive": pos, "p_negative": neg, "p_neutral": neu,
                        "model": "finbert",
                    })
            return results
        except Exception:
            pass

    if _VADER:
        return [_vader_score(t) for t in texts]

    return [_keyword_score_dict(t) for t in texts]


# ── Aggregation ───────────────────────────────────────────────────────────────

def score_posts_df(posts_df: pd.DataFrame, prefer_finbert: bool = True) -> pd.DataFrame:
    """
    Add sentiment and emotion columns to a posts DataFrame.

    Input:  DataFrame with 'title' and 'selftext' columns
    Output: Same DataFrame + compound, sentiment_label, emotion, confidence
    """
    if posts_df.empty:
        return posts_df

    df   = posts_df.copy()
    # Combine title (weighted 2×) + selftext for richer signal
    texts = (df["title"].fillna("") + " " + df["title"].fillna("") + " "
             + df["selftext"].fillna("")).tolist()

    scores = score_texts(texts, prefer_finbert=prefer_finbert)
    df["compound"]        = [s["compound"]    for s in scores]
    df["sentiment_label"] = [s["label"]       for s in scores]
    df["confidence"]      = [s["confidence"]  for s in scores]
    df["p_positive"]      = [s["p_positive"]  for s in scores]
    df["p_negative"]      = [s["p_negative"]  for s in scores]
    df["emotion"]         = classify_emotions_batch(texts)
    df["score_model"]     = scores[0]["model"] if scores else "unknown"

    return df


def aggregate_daily(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate scored posts to daily sentiment metrics.

    Returns DataFrame indexed by date with columns:
      n_posts, mean_compound, pct_bullish, pct_bearish, pct_neutral,
      pct_euphoria, pct_fomo, pct_fud, pct_capitulation, pct_cautious,
      weighted_compound (weighted by score)
    """
    if scored_df.empty or "compound" not in scored_df.columns:
        return pd.DataFrame()

    df         = scored_df.copy()
    df["date"] = df["created_utc"].dt.date

    def _agg(g):
        n = len(g)
        w = np.clip(np.log1p(g["score"].clip(lower=0)), 0, None)
        w_sum = w.sum()
        weighted = float((g["compound"] * w).sum() / w_sum) if w_sum > 0 else float(g["compound"].mean())

        emotions = Counter(g["emotion"])
        return pd.Series({
            "n_posts":          n,
            "mean_compound":    float(g["compound"].mean()),
            "weighted_compound": weighted,
            "pct_bullish":      (g["sentiment_label"] == "bullish").mean() * 100,
            "pct_bearish":      (g["sentiment_label"] == "bearish").mean() * 100,
            "pct_neutral":      (g["sentiment_label"] == "neutral").mean() * 100,
            "pct_euphoria":     emotions.get("euphoria",        0) / n * 100,
            "pct_fomo":         emotions.get("fomo",            0) / n * 100,
            "pct_fud":          emotions.get("fud",             0) / n * 100,
            "pct_capitulation": emotions.get("capitulation",    0) / n * 100,
            "pct_cautious":     emotions.get("cautious_optimism", 0) / n * 100,
        })

    daily = df.groupby("date").apply(_agg)
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def sentiment_label_from_compound(compound: float) -> str:
    """Map a compound score to a human-readable sentiment label."""
    if compound > 0.45:
        return "VERY BULLISH"
    if compound > 0.20:
        return "CAUTIOUSLY OPTIMISTIC"
    if compound > 0.05:
        return "MILDLY BULLISH"
    if compound < -0.45:
        return "EXTREME FEAR / CAPITULATION"
    if compound < -0.20:
        return "BEARISH"
    if compound < -0.05:
        return "MILDLY BEARISH"
    return "NEUTRAL"
