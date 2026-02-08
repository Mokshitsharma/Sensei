# src/domain/signals.py
from typing import Dict, List


def generate_signal(
    indicators: Dict[str, float],
    patterns: List[dict],
    fundamentals: Dict[str, float],
) -> Dict[str, object]:
    score = 0
    reasons: List[str] = []

    # ---------------------------
    # RSI
    # ---------------------------
    rsi = indicators["rsi"]
    if rsi < 30:
        score += 2
        reasons.append("RSI indicates oversold conditions")
    elif rsi > 70:
        score -= 2
        reasons.append("RSI indicates overbought conditions")

    # ---------------------------
    # MACD
    # ---------------------------
    if indicators["macd"] > indicators["macd_signal"]:
        score += 1
        reasons.append("MACD is bullish")
    else:
        score -= 1
        reasons.append("MACD is bearish")

    # ---------------------------
    # Patterns
    # ---------------------------
    for event in patterns:
        if event["type"] == "golden_cross":
            score += 2
            reasons.append("Golden Cross detected")
        elif event["type"] == "death_cross":
            score -= 2
            reasons.append("Death Cross detected")
        elif event["type"] == "breakout":
            score += 1
            reasons.append("Price breakout detected")

    # ---------------------------
    # Fundamentals
    # ---------------------------
    if fundamentals.get("pe") and fundamentals["pe"] < 25:
        score += 1
        reasons.append("Reasonable PE valuation")

    if fundamentals.get("roe") and fundamentals["roe"] > 15:
        score += 1
        reasons.append("Strong ROE")

    # ---------------------------
    # Final Signal
    # ---------------------------
    if score >= 4:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    # ---------------------------
    # Confidence (bounded)
    # ---------------------------
    confidence = min(90, max(30, abs(score) * 15))

    return {
        "signal": signal,
        "score": score,
        "confidence": confidence,
        "reasons": reasons,
    }
