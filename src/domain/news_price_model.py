# src/domain/news_price_model.py
"""
News-driven Price Prediction Model.

Logic:
  1. Compute a "news impact magnitude" from the weighted sentiment score
     and the mix of high-impact (Earnings/Regulatory) vs low-impact categories.
  2. Scale the magnitude by ATR (volatility): the same sentiment score moves
     a high-ATR stock more than a low-ATR stock.
  3. Build a predicted price range (low, high) and a point estimate.
  4. Assign a confidence band based on how consistent the headlines are
     (all bullish = high confidence; mixed = low confidence).

This is a heuristic model, not an ML model. It is transparent and
explainable — every number can be traced back to a news item.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Base % move per unit of weighted sentiment score (tuned to NSE large caps)
_BASE_MOVE_PCT = 0.012       # 1.2 % per 1.0 sentiment unit

# ATR multiplier: impact = sentiment_magnitude × (atr_pct / atr_baseline)
_ATR_BASELINE = 0.015        # 1.5 % daily ATR is "normal" for NIFTY 50

# Horizon labels
_HORIZONS = {
    "1d": "1 trading day",
    "3d": "1–3 trading days",
    "5d": "3–5 trading days",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_news_price_impact(
    current_price: float,
    news_result: Dict,
    atr: float,
    horizon: str = "3d",
) -> Dict:
    """
    Predict expected price range driven by current news sentiment.

    Args:
        current_price : latest close price (₹)
        news_result   : output of analyze_news_sentiment()
        atr           : current ATR (absolute ₹ value)
        horizon       : "1d" | "3d" | "5d"

    Returns dict:
        predicted_price   : float  (point estimate)
        price_low         : float  (conservative estimate)
        price_high        : float  (optimistic estimate)
        expected_move_pct : float  (signed %, e.g. +1.8 or -2.3)
        direction         : UP | DOWN | FLAT
        confidence        : HIGH | MEDIUM | LOW
        horizon_label     : str
        explanation       : plain-English paragraph
    """
    weighted_score = news_result.get("weighted_score", 0.0)
    bull_count     = news_result.get("bull_count", 0)
    bear_count     = news_result.get("bear_count", 0)
    neutral_count  = news_result.get("neutral_count", 0)
    details        = news_result.get("details", [])

    total = bull_count + bear_count + neutral_count
    if total == 0 or current_price <= 0:
        return _flat_prediction(current_price, horizon)

    # ── Step 1: base move % ─────────────────────────────────────────────────
    # Scale by how "loud" the high-impact news is
    high_impact_types = {"Earnings", "Regulatory", "Management"}
    high_impact_items = [d for d in details if d.get("impact_type") in high_impact_types]
    high_impact_boost  = 1.0 + 0.3 * (len(high_impact_items) / max(total, 1))

    atr_pct     = atr / current_price if current_price > 0 else _ATR_BASELINE
    atr_scaling = atr_pct / _ATR_BASELINE   # >1 means more volatile stock

    # Horizon scaling: longer horizon = larger potential move
    horizon_scale = {"1d": 0.6, "3d": 1.0, "5d": 1.4}.get(horizon, 1.0)

    base_move_pct = (
        weighted_score
        * _BASE_MOVE_PCT
        * high_impact_boost
        * atr_scaling
        * horizon_scale
    )

    # ── Step 2: uncertainty band ─────────────────────────────────────────────
    # If all headlines agree → narrow band; mixed → wide band
    agreement = abs(bull_count - bear_count) / max(total, 1)
    band_width_pct = atr_pct * (1.5 - agreement)   # wide when mixed

    # ── Step 3: predicted prices ─────────────────────────────────────────────
    predicted_price = round(current_price * (1 + base_move_pct), 2)
    price_low       = round(current_price * (1 + base_move_pct - band_width_pct), 2)
    price_high      = round(current_price * (1 + base_move_pct + band_width_pct), 2)

    expected_move_pct = round(base_move_pct * 100, 2)

    # ── Step 4: direction + confidence ───────────────────────────────────────
    if abs(expected_move_pct) < 0.3:
        direction = "FLAT"
    elif expected_move_pct > 0:
        direction = "UP"
    else:
        direction = "DOWN"

    if agreement > 0.7 and abs(weighted_score) > 0.3:
        confidence = "HIGH"
    elif agreement > 0.4 or abs(weighted_score) > 0.15:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    explanation = _build_explanation(
        direction, expected_move_pct, confidence,
        bull_count, bear_count, neutral_count,
        high_impact_items, current_price, predicted_price,
        price_low, price_high, horizon,
    )

    return {
        "predicted_price":    predicted_price,
        "price_low":          price_low,
        "price_high":         price_high,
        "expected_move_pct":  expected_move_pct,
        "direction":          direction,
        "confidence":         confidence,
        "horizon_label":      _HORIZONS.get(horizon, horizon),
        "explanation":        explanation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flat_prediction(price: float, horizon: str) -> Dict:
    return {
        "predicted_price":   round(price, 2),
        "price_low":         round(price, 2),
        "price_high":        round(price, 2),
        "expected_move_pct": 0.0,
        "direction":         "FLAT",
        "confidence":        "LOW",
        "horizon_label":     _HORIZONS.get(horizon, horizon),
        "explanation":       "Insufficient news data to generate a price prediction.",
    }


def _build_explanation(
    direction, move_pct, confidence,
    bull, bear, neutral,
    high_impact_items, current_price, predicted, low, high, horizon,
) -> str:
    total = bull + bear + neutral
    horizon_label = _HORIZONS.get(horizon, horizon)

    dir_phrase = {
        "UP":   f"an upward move of approximately {abs(move_pct):.1f}%",
        "DOWN": f"a downward move of approximately {abs(move_pct):.1f}%",
        "FLAT": "flat price action",
    }.get(direction, "mixed movement")

    conf_phrase = {
        "HIGH":   "This is a high-confidence signal",
        "MEDIUM": "This is a moderate-confidence signal",
        "LOW":    "This is a low-confidence signal",
    }.get(confidence, "Confidence is uncertain")

    hi_names = list({d.get("impact_type", "") for d in high_impact_items})
    hi_phrase = (
        f" High-impact {', '.join(hi_names)} news is amplifying the signal."
        if hi_names else ""
    )

    return (
        f"Based on {total} news headlines ({bull} bullish, {bear} bearish, {neutral} neutral), "
        f"the news-driven model projects {dir_phrase} over {horizon_label}.{hi_phrase} "
        f"Predicted price range: ₹{low:,.2f} – ₹{high:,.2f} (point estimate: ₹{predicted:,.2f}). "
        f"{conf_phrase} — {'headlines largely agree' if confidence == 'HIGH' else 'headlines are mixed' if confidence == 'LOW' else 'moderate headline agreement'}. "
        f"Note: this forecast is based on sentiment only and should be combined with technical signals."
    )
