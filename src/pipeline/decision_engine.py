# src/pipeline/decision_engine.py

from typing import Dict


def make_final_decision(
    signals: Dict[str, float],
    news_sentiment: float,
) -> Dict[str, object]:
    """
    Combine ML, DL, RL, regime, and news signals
    into a final BUY / SELL / HOLD decision.
    """

    score = 0.0
    explanation = []

    # -----------------------------
    # ML probability
    # -----------------------------
    if signals["ml_prob_up"] > 0.6:
        score += 1
        explanation.append("ML model is bullish")
    elif signals["ml_prob_up"] < 0.4:
        score -= 1
        explanation.append("ML model is bearish")

    # -----------------------------
    # LSTM / TCN returns
    # -----------------------------
    if signals["lstm_return"] > 0:
        score += 1
        explanation.append("LSTM predicts positive return")
    else:
        score -= 1
        explanation.append("LSTM predicts negative return")

    if signals["tcn_return"] > 0:
        score += 0.5
        explanation.append("TCN confirms upside")

    # -----------------------------
    # Regime
    # -----------------------------
    if signals["regime"] == "BULL":
        score += 1
        explanation.append("Market regime is bullish")
    elif signals["regime"] == "BEAR":
        score -= 1
        explanation.append("Market regime is bearish")

    # -----------------------------
    # PPO agent
    # -----------------------------
    if signals["ppo_action"] == "BUY":
        score += 1
        explanation.append("RL agent suggests BUY")
    elif signals["ppo_action"] == "SELL":
        score -= 1
        explanation.append("RL agent suggests SELL")

    # -----------------------------
    # News sentiment
    # -----------------------------
    if news_sentiment > 0.2:
        score += 1
        explanation.append("Positive news sentiment")
    elif news_sentiment < -0.2:
        score -= 1
        explanation.append("Negative news sentiment")

    # -----------------------------
    # Final Decision
    # -----------------------------
    if score >= 2:
        action = "BUY"
    elif score <= -2:
        action = "SELL"
    else:
        action = "HOLD"

    confidence = min(abs(score) / 5, 1.0)

    return {
        "action": action,
        "confidence": confidence,
        "score": score,
        "explanation": " | ".join(explanation),
    }
