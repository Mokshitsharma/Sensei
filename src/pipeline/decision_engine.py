# src/pipeline/decision_engine.py
"""
Decision Engine — final BUY/SELL/HOLD verdict with enriched explainability.

Changes vs original:
  • Accepts optional `shap_values` and `feature_values` kwargs
  • Calls the Market Narrator to produce a structured plain-English report
  • Returns `shap_ranked`, `narrative` in the output dict
"""

from typing import Dict, Optional, List

from src.explainability.narrator import build_narrative


def make_final_decision(
    signals: Dict,
    news_sentiment: float,
    shap_values: Optional[Dict[str, float]] = None,
    feature_values: Optional[Dict[str, float]] = None,
    company: str = "the stock",
) -> Dict:

    """
    Aggregate all model signals into a final trading decision.

    Args:
        signals:        Output of signal_pipeline
        news_sentiment: Scalar sentiment score from NLP layer
        shap_values:    {feature: shap_value} for the latest ML prediction (optional)
        feature_values: {feature: raw_value} for the latest bar (optional)
        company:        Human-readable company name for narrative generation

    Returns dict with keys:
        action, confidence, score, explanation,
        shap_ranked, narrative
    """

    score = 0.0
    explanation = []

    ml_prob   = signals.get("ml_prob_up", 0.5)
    lstm_ret  = signals.get("lstm_return", 0.0)
    tcn_ret   = signals.get("tcn_return", 0.0)
    regime    = signals.get("regime", "NEUTRAL")
    ppo_act   = signals.get("ppo_action", "HOLD")

    # ── ML Ensemble ──────────────────────────────────────────────────────────
    if ml_prob > 0.6:
        score += 1
        explanation.append(f"ML model is bullish ({ml_prob*100:.1f}% UP probability)")
    elif ml_prob < 0.4:
        score -= 1
        explanation.append(f"ML model is bearish ({ml_prob*100:.1f}% UP probability)")
    else:
        explanation.append(f"ML model is neutral ({ml_prob*100:.1f}% UP probability)")

    # ── LSTM ─────────────────────────────────────────────────────────────────
    if lstm_ret > 0:
        score += 1
        explanation.append(f"LSTM forecasts +{lstm_ret*100:.2f}% over 5 days")
    else:
        score -= 1
        explanation.append(f"LSTM forecasts {lstm_ret*100:.2f}% over 5 days")

    # ── TCN ──────────────────────────────────────────────────────────────────
    if tcn_ret > 0:
        score += 0.5
        explanation.append(f"TCN confirms upside (+{tcn_ret*100:.2f}%)")
    else:
        explanation.append(f"TCN forecasts downside ({tcn_ret*100:.2f}%)")

    # ── Market Regime ────────────────────────────────────────────────────────
    if regime == "BULL":
        score += 1
        explanation.append("HMM detects BULL market regime")
    elif regime == "BEAR":
        score -= 1
        explanation.append("HMM detects BEAR market regime")
    else:
        explanation.append("HMM regime is NEUTRAL")

    # ── PPO Agent ────────────────────────────────────────────────────────────
    if ppo_act == "BUY":
        score += 1
        explanation.append("RL agent recommends BUY")
    elif ppo_act == "SELL":
        score -= 1
        explanation.append("RL agent recommends SELL")
    else:
        explanation.append("RL agent recommends HOLD")

    # ── News Sentiment ───────────────────────────────────────────────────────
    if news_sentiment > 0.2:
        score += 1
        explanation.append(f"Positive news sentiment ({news_sentiment:.2f})")
    elif news_sentiment < -0.2:
        score -= 1
        explanation.append(f"Negative news sentiment ({news_sentiment:.2f})")
    else:
        explanation.append(f"Neutral news sentiment ({news_sentiment:.2f})")

    # ── Final Decision ───────────────────────────────────────────────────────
    if score >= 2:
        action = "BUY"
    elif score <= -2:
        action = "SELL"
    else:
        action = "HOLD"

    confidence = min(abs(score) / 5.0, 1.0)

    # ── SHAP Ranking ─────────────────────────────────────────────────────────
    shap_ranked: List[Dict] = []
    if shap_values:
        from src.ml.shap_explain import rank_features_by_impact
        shap_ranked = rank_features_by_impact(shap_values)

    # ── Human-readable Narrative ─────────────────────────────────────────────
    decision_so_far = {
        "action":     action,
        "confidence": confidence,
        "score":      score,
    }

    narrative = build_narrative(
        signals=signals,
        decision=decision_so_far,
        feature_values=feature_values or {},
        shap_ranked=shap_ranked if shap_ranked else None,
        company=company,
        news_sentiment=news_sentiment,
    )

    return {
        "action":      action,
        "confidence":  confidence,
        "score":       score,
        "explanation": " | ".join(explanation) or "Mixed signals",
        "shap_ranked": shap_ranked,
        "narrative":   narrative,
    }
