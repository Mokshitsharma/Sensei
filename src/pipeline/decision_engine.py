# src/pipeline/decision_engine.py

from typing import Dict


def make_final_decision(
    signals: Dict[str, float],
    news_sentiment: float,
) -> Dict[str, object]:

    score = 0.0
    explanation = []

    ml_prob = signals.get("ml_prob_up", 0.5)
    lstm_return = signals.get("lstm_return", 0.0)
    tcn_return = signals.get("tcn_return", 0.0)
    regime = signals.get("regime", "NEUTRAL")
    ppo_action = signals.get("ppo_action", "HOLD")

    # ML
    if ml_prob > 0.6:
        score += 1
        explanation.append("ML model is bullish")
    elif ml_prob < 0.4:
        score -= 1
        explanation.append("ML model is bearish")

    # LSTM
    if lstm_return > 0:
        score += 1
        explanation.append("LSTM predicts positive return")
    else:
        score -= 1
        explanation.append("LSTM predicts negative return")

    # TCN
    if tcn_return > 0:
        score += 0.5
        explanation.append("TCN confirms upside")

    # Regime
    if regime == "BULL":
        score += 1
        explanation.append("Market regime is bullish")
    elif regime == "BEAR":
        score -= 1
        explanation.append("Market regime is bearish")

    # PPO
    if ppo_action == "BUY":
        score += 1
        explanation.append("RL agent suggests BUY")
    elif ppo_action == "SELL":
        score -= 1
        explanation.append("RL agent suggests SELL")

    # News
    if news_sentiment > 0.2:
        score += 1
        explanation.append("Positive news sentiment")
    elif news_sentiment < -0.2:
        score -= 1
        explanation.append("Negative news sentiment")

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
        "explanation": " | ".join(explanation) or "Mixed signals",
    }
