# src/domain/signals.py

from typing import Dict, List


def generate_signal(
    indicators: Dict[str, float],
    patterns: List[Dict],
    fundamentals: Dict[str, float],
) -> Dict[str, object]:
    """
    Generate rule-based trading signal.

    Returns:
        {
            "action": BUY | SELL | HOLD,
            "confidence": float (0-1),
            "explanation": str
        }
    """

    score = 0.0
    reasons = []

    # -----------------------------
    # Indicator-based rules
    # -----------------------------
    rsi = indicators.get("rsi", 50)

    if rsi < 30:
        score += 1.0
        reasons.append("RSI oversold")

    elif rsi > 70:
        score -= 1.0
        reasons.append("RSI overbought")

    macd = indicators.get("macd", 0)
    macd_signal = indicators.get("macd_signal", 0)

    if macd > macd_signal:
        score += 0.5
        reasons.append("MACD bullish crossover")

    else:
        score -= 0.5
        reasons.append("MACD bearish crossover")

    # -----------------------------
    # Pattern-based rules
    # -----------------------------
    for p in patterns:
        p_type = p["type"]
        strength = p.get("strength", 0.5)

        if p_type in ("golden_cross", "breakout", "bullish_engulfing"):
            score += strength
            reasons.append(p_type.replace("_", " ").title())

        if p_type in ("death_cross", "breakdown", "bearish_engulfing"):
            score -= strength
            reasons.append(p_type.replace("_", " ").title())

    # -----------------------------
    # Fundamental sanity checks
    # -----------------------------
    price = fundamentals.get("current_price", 0)
    high_52 = fundamentals.get("52_week_high", price)
    low_52 = fundamentals.get("52_week_low", price)

    if price <= low_52 * 1.05:
        score += 0.5
        reasons.append("Near 52-week low")

    if price >= high_52 * 0.95:
        score -= 0.5
        reasons.append("Near 52-week high")

    debt_to_equity = fundamentals.get("debt_to_equity", 0)

    if debt_to_equity > 2:
        score -= 0.5
        reasons.append("High leverage")

    roe = fundamentals.get("roe", 0)

    if roe > 0.15:
        score += 0.5
        reasons.append("Strong ROE")

    # -----------------------------
    # Final decision
    # -----------------------------
    if score >= 1.5:
        action = "BUY"
    elif score <= -1.5:
        action = "SELL"
    else:
        action = "HOLD"

    confidence = min(abs(score) / 3.0, 1.0)

    explanation = "; ".join(reasons) if reasons else "No strong signals"

    return {
        "action": action,
        "confidence": round(confidence, 2),
        "score": round(score, 2),
        "explanation": explanation,
    }