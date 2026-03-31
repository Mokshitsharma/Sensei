# src/data/news.py
"""
News Engine — fetches, categorises, and scores news for a stock.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import feedparser


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment — try FinBERT first, fall back to keyword VADER
# ─────────────────────────────────────────────────────────────────────────────

_sentiment_pipeline = None


def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        try:
            _sentiment_pipeline = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=1,
            )
        except Exception:
            _sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )
        return _sentiment_pipeline
    except Exception:
        return None


def _vader_score(text: str) -> Dict:
    text_lower = text.lower()
    bull_words = [
        "surge", "rally", "gain", "profit", "beat", "upgrade", "buy",
        "growth", "record", "high", "strong", "positive", "rise", "jump",
        "outperform", "bullish", "dividend", "buyback", "expansion",
    ]
    bear_words = [
        "fall", "drop", "loss", "miss", "downgrade", "sell", "decline",
        "weak", "low", "negative", "crash", "plunge", "bearish", "layoff",
        "recall", "fine", "penalty", "fraud", "default", "concern", "risk",
    ]
    bull_hits = sum(1 for w in bull_words if w in text_lower)
    bear_hits = sum(1 for w in bear_words if w in text_lower)
    if bull_hits > bear_hits:
        return {"label": "POSITIVE", "score": min(0.5 + bull_hits * 0.1, 0.95)}
    if bear_hits > bull_hits:
        return {"label": "NEGATIVE", "score": min(0.5 + bear_hits * 0.1, 0.95)}
    return {"label": "NEUTRAL", "score": 0.5}


def _score_headline(text: str) -> Dict:
    pipe = _get_sentiment_pipeline()
    if pipe is None:
        return _vader_score(text)
    try:
        result = pipe(text[:512])[0]
        if isinstance(result, list):
            result = result[0]
        label = result["label"].upper()
        if label not in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
            label = ("POSITIVE" if label.startswith("POS")
                     else "NEGATIVE" if label.startswith("NEG") else "NEUTRAL")
        return {"label": label, "score": float(result["score"])}
    except Exception:
        return _vader_score(text)


# ─────────────────────────────────────────────────────────────────────────────
# Impact type categorisation
# ─────────────────────────────────────────────────────────────────────────────

_IMPACT_KEYWORDS: Dict[str, List[str]] = {
    "Earnings": [
        "earnings", "profit", "revenue", "quarterly", "results", "eps",
        "net income", "q1", "q2", "q3", "q4", "fy", "guidance", "outlook",
        "forecast", "beat", "miss", "dividend",
    ],
    "Regulatory": [
        "sebi", "rbi", "nse", "bse", "penalty", "fine", "compliance",
        "regulation", "ban", "investigation", "probe", "audit",
        "court", "lawsuit", "legal",
    ],
    "Management": [
        "ceo", "cfo", "director", "board", "appoint", "resign", "management",
        "leadership", "stake", "promoter", "buyback", "open offer",
    ],
    "Macro": [
        "rate", "inflation", "gdp", "economy", "fiscal", "budget", "policy",
        "interest rate", "repo", "rupee", "dollar", "crude", "oil", "global",
    ],
    "Sector": [
        "sector", "industry", "peer", "rival", "competitor", "market share",
        "ipo", "nifty", "sensex", "index",
    ],
}

_IMPACT_WEIGHTS: Dict[str, float] = {
    "Earnings":   1.5,
    "Regulatory": 1.3,
    "Management": 1.2,
    "Macro":      0.9,
    "Sector":     0.8,
    "General":    0.6,
}


def _classify_impact(text: str) -> str:
    text_lower = text.lower()
    for category, keywords in _IMPACT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "General"


# ─────────────────────────────────────────────────────────────────────────────
# Fetching
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_news(
    company: str,
    ticker: str = "",
    max_items: int = 10,
) -> List[Dict]:
    results = []
    queries = [company.replace(" ", "+") + "+stock+India"]
    if ticker:
        clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
        if clean_ticker.lower() not in company.lower():
            queries.append(clean_ticker + "+NSE")

    seen_titles = set()

    for query in queries:
        url = (
            f"https://news.google.com/rss/search?q={query}"
            f"&hl=en-IN&gl=IN&ceid=IN:en"
        )
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = _clean_text(entry.get("title", ""))
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                try:
                    ps = entry.get("published_parsed")
                    published = (
                        datetime(*ps[:6], tzinfo=timezone.utc).strftime("%d %b %Y, %H:%M")
                        if ps else "—"
                    )
                except Exception:
                    published = "—"

                results.append({
                    "headline":  title,
                    "source":    entry.get("source", {}).get("title", "News"),
                    "published": published,
                    "url":       entry.get("link", ""),
                })
                if len(results) >= max_items:
                    break
        except Exception:
            continue
        if len(results) >= max_items:
            break

    return results[:max_items]


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment aggregation
# ─────────────────────────────────────────────────────────────────────────────

def analyze_news_sentiment(news_items: List[Dict]) -> Dict:
    if not news_items:
        return _empty_sentiment()

    details = []
    raw_scores = []
    weighted_scores = []
    bull_count = bear_count = neutral_count = 0

    for item in news_items:
        headline    = item["headline"]
        impact_type = _classify_impact(headline)
        weight      = _IMPACT_WEIGHTS.get(impact_type, 0.6)
        sent        = _score_headline(headline)

        label = sent["label"]
        conf  = round(float(sent["score"]), 3)

        if label == "POSITIVE":
            signed = conf
            bull_count += 1
        elif label == "NEGATIVE":
            signed = -conf
            bear_count += 1
        else:
            signed = 0.0
            neutral_count += 1

        raw_scores.append(signed)
        weighted_scores.append(signed * weight)

        details.append({
            "headline":     headline,
            "source":       item.get("source", "—"),
            "published":    item.get("published", "—"),
            "url":          item.get("url", ""),
            "label":        label,
            "confidence":   conf,
            "impact_type":  impact_type,
            "weight":       weight,
            "signed_score": round(signed * weight, 4),
        })

    avg_raw      = sum(raw_scores) / len(raw_scores)
    avg_weighted = sum(weighted_scores) / len(weighted_scores)

    bullish_items = [d for d in details if d["label"] == "POSITIVE"]
    bearish_items = [d for d in details if d["label"] == "NEGATIVE"]

    top_bullish = max(bullish_items, key=lambda x: x["signed_score"], default=None)
    top_bearish = min(bearish_items, key=lambda x: x["signed_score"], default=None)

    return {
        "sentiment_score": round(avg_raw, 3),
        "weighted_score":  round(avg_weighted, 3),
        "bull_count":      bull_count,
        "bear_count":      bear_count,
        "neutral_count":   neutral_count,
        "top_bullish":     top_bullish,
        "top_bearish":     top_bearish,
        "summary":         _build_summary(avg_weighted, bull_count, bear_count, neutral_count),
        "details":         details,
    }


def _build_summary(score: float, bull: int, bear: int, neutral: int) -> str:
    total = bull + bear + neutral
    if total == 0:
        return "No news found."
    if score > 0.4:
        tone, impact = "strongly bullish", "likely to support upward price movement"
    elif score > 0.15:
        tone, impact = "mildly bullish", "may provide a modest tailwind"
    elif score < -0.4:
        tone, impact = "strongly bearish", "likely to pressure the stock price downward"
    elif score < -0.15:
        tone, impact = "mildly bearish", "may act as a mild headwind"
    else:
        tone, impact = "neutral", "unlikely to significantly move the price"
    return (
        f"News sentiment is {tone} ({bull} bullish, {bear} bearish, {neutral} neutral "
        f"out of {total} headlines). Recent coverage is {impact}."
    )


def _empty_sentiment() -> Dict:
    return {
        "sentiment_score": 0.0,
        "weighted_score":  0.0,
        "bull_count":      0,
        "bear_count":      0,
        "neutral_count":   0,
        "top_bullish":     None,
        "top_bearish":     None,
        "summary":         "No recent news found for this stock.",
        "details":         [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_news_signal(
    company: str,
    ticker: str = "",
    max_items: int = 10,
) -> Dict:
    items  = fetch_news(company, ticker=ticker, max_items=max_items)
    result = analyze_news_sentiment(items)
    result["headlines"] = [d["headline"] for d in result["details"]]
    return result
