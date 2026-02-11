# src/data/news.py

from typing import List, Dict
import feedparser
import re

from transformers import pipeline


_sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)


def _clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_news(
    company: str,
    max_items: int = 5,
) -> List[str]:
    """
    Fetch recent news headlines related to a company.
    Uses Google News RSS (no API key).
    """
    query = company.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}+stock+market"

    feed = feedparser.parse(url)

    headlines = []
    for entry in feed.entries[:max_items]:
        headlines.append(_clean_text(entry.title))

    return headlines


def analyze_news_sentiment(
    headlines: List[str],
) -> Dict[str, object]:
    """
    Analyze sentiment of news headlines.

    Returns:
        {
            "sentiment_score": float (-1 to +1),
            "summary": str,
            "details": List[Dict]
        }
    """
    if not headlines:
        return {
            "sentiment_score": 0.0,
            "summary": "No recent significant news detected.",
            "details": [],
        }

    scores = []
    details = []

    for h in headlines:
        result = _sentiment_model(h)[0]
        score = result["score"]
        signed_score = score if result["label"] == "POSITIVE" else -score

        scores.append(signed_score)
        details.append(
            {
                "headline": h,
                "label": result["label"],
                "confidence": round(score, 3),
            }
        )

    avg_score = sum(scores) / len(scores)

    if avg_score > 0.2:
        summary = "Recent news is largely positive and may support upward price movement."
    elif avg_score < -0.2:
        summary = "Recent news is largely negative and may pressure the stock price."
    else:
        summary = "Recent news sentiment is neutral with limited expected price impact."

    return {
        "sentiment_score": round(avg_score, 3),
        "summary": summary,
        "details": details,
    }


def get_news_signal(company: str) -> Dict[str, object]:
    """
    End-to-end news sentiment pipeline.
    """
    headlines = fetch_news(company)
    return analyze_news_sentiment(headlines)