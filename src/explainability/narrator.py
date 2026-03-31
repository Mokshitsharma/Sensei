# src/explainability/narrator.py
"""
Market Narrator — converts raw signals + SHAP feature impacts into
plain-English market commentary that a retail investor can read.

Design principles:
  • Every sentence is grounded in an actual signal value (no hallucination)
  • Language scales with magnitude (e.g. "mildly overbought" vs "extremely overbought")
  • Narrative is structured: Trend → Momentum → Volatility → Regime → AI Consensus
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────────

def build_narrative(
    signals: Dict,
    decision: Dict,
    feature_values: Dict[str, float],
    shap_ranked: Optional[List[Dict]] = None,
    company: str = "the stock",
    news_sentiment: float = 0.0,
) -> Dict[str, str]:
    """
    Build a structured, human-readable market narrative.

    Args:
        signals:        Output of signal_pipeline (ml_prob_up, lstm_return, etc.)
        decision:       Output of decision_engine (action, confidence, score)
        feature_values: Latest raw feature values {feature_name: value}
        shap_ranked:    Output of rank_features_by_impact (optional)
        company:        Human-readable company name
        news_sentiment: Scalar sentiment score

    Returns:
        Dict with keys:
            "headline"    – one-liner verdict
            "trend"       – EMA / regime paragraph
            "momentum"    – RSI / MACD paragraph
            "volatility"  – ATR / vol paragraph
            "ai_models"   – LSTM / TCN / PPO / ML summary paragraph
            "shap_story"  – top SHAP drivers in plain English
            "news"        – news sentiment line
            "summary"     – final 2-sentence synthesis
    """

    action    = decision.get("action", "HOLD")
    conf_pct  = round(decision.get("confidence", 0) * 100, 1)
    score     = decision.get("score", 0)
    regime    = signals.get("regime", "NEUTRAL")

    rsi_norm      = feature_values.get("rsi_norm", 0.5)
    rsi           = rsi_norm * 100
    ema_spread    = feature_values.get("ema_spread", 0.0)
    macd_diff     = feature_values.get("macd_diff", 0.0)
    atr_pct       = feature_values.get("atr_pct", 0.0)
    volatility_10 = feature_values.get("volatility_10", 0.0)
    return_1      = feature_values.get("return_1", 0.0)
    return_5      = feature_values.get("return_5", 0.0)
    range_pos     = feature_values.get("range_position", 0.5)

    ml_prob  = signals.get("ml_prob_up", 0.5)
    lstm_ret = signals.get("lstm_return", 0.0)
    tcn_ret  = signals.get("tcn_return", 0.0)
    ppo_act  = signals.get("ppo_action", "HOLD")

    narrative = {}

    # ── Headline ─────────────────────────────────────────────────────────────
    narrative["headline"] = _headline(company, action, conf_pct, regime)

    # ── Trend ────────────────────────────────────────────────────────────────
    narrative["trend"] = _trend_paragraph(
        company, ema_spread, regime, range_pos, return_5
    )

    # ── Momentum ─────────────────────────────────────────────────────────────
    narrative["momentum"] = _momentum_paragraph(rsi, macd_diff, return_1)

    # ── Volatility ───────────────────────────────────────────────────────────
    narrative["volatility"] = _volatility_paragraph(atr_pct, volatility_10)

    # ── AI Models ────────────────────────────────────────────────────────────
    narrative["ai_models"] = _ai_models_paragraph(
        ml_prob, lstm_ret, tcn_ret, ppo_act
    )

    # ── SHAP Story ───────────────────────────────────────────────────────────
    if shap_ranked:
        narrative["shap_story"] = _shap_paragraph(shap_ranked, feature_values)
    else:
        narrative["shap_story"] = (
            "SHAP feature attribution is unavailable for this model type."
        )

    # ── News ─────────────────────────────────────────────────────────────────
    narrative["news"] = _news_paragraph(company, news_sentiment)

    # ── Summary ──────────────────────────────────────────────────────────────
    narrative["summary"] = _summary_paragraph(
        company, action, conf_pct, score, regime
    )

    return narrative


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────

def _headline(company: str, action: str, conf_pct: float, regime: str) -> str:
    action_phrases = {
        "BUY":  "signals a buying opportunity",
        "SELL": "signals a selling opportunity",
        "HOLD": "recommends holding your position",
    }
    regime_phrase = {
        "BULL": "in a bullish market environment",
        "BEAR": "against a bearish market backdrop",
    }.get(regime, "in a mixed market environment")

    verb = action_phrases.get(action, "returns a mixed signal")
    return (
        f"Sensei AI {verb} for {company} ({conf_pct}% confidence) "
        f"{regime_phrase}."
    )


def _trend_paragraph(
    company: str,
    ema_spread: float,
    regime: str,
    range_pos: float,
    return_5: float,
) -> str:
    # EMA spread interpretation
    if ema_spread > 0.02:
        ema_line = (
            f"The 20-day EMA is trading well above the 50-day EMA "
            f"(spread: {ema_spread*100:.2f}%), indicating a strong uptrend."
        )
    elif ema_spread > 0.005:
        ema_line = (
            f"The short-term EMA is slightly above the long-term EMA "
            f"(spread: {ema_spread*100:.2f}%), suggesting a mild upward bias."
        )
    elif ema_spread < -0.02:
        ema_line = (
            f"The 20-day EMA is trading below the 50-day EMA "
            f"(spread: {ema_spread*100:.2f}%), signalling a downtrend."
        )
    elif ema_spread < -0.005:
        ema_line = (
            f"The short-term EMA has slipped under the long-term EMA "
            f"(spread: {ema_spread*100:.2f}%), hinting at bearish pressure."
        )
    else:
        ema_line = (
            "The 20-day and 50-day EMAs are nearly flat against each other, "
            "pointing to sideways price action."
        )

    # Price range positioning
    if range_pos > 0.8:
        range_line = (
            f"{company} is trading near its 20-day high "
            f"({range_pos*100:.0f}th percentile of its range), "
            "reflecting strong demand."
        )
    elif range_pos < 0.2:
        range_line = (
            f"{company} is trading near its 20-day low "
            f"({range_pos*100:.0f}th percentile of its range), "
            "suggesting seller control."
        )
    else:
        range_line = (
            f"{company} is positioned in the middle of its 20-day range "
            f"({range_pos*100:.0f}th percentile), showing balanced price action."
        )

    # 5-day return context
    ret5_pct = return_5 * 100
    if abs(ret5_pct) < 0.5:
        ret_line = "Price has been largely unchanged over the past week."
    elif ret5_pct > 0:
        ret_line = f"The stock has gained {ret5_pct:.1f}% over the past 5 trading days."
    else:
        ret_line = f"The stock has declined {abs(ret5_pct):.1f}% over the past 5 trading days."

    regime_line = {
        "BULL": "The HMM regime detector classifies the current macro environment as BULLISH.",
        "BEAR": "The HMM regime detector classifies the current macro environment as BEARISH — caution is warranted.",
    }.get(regime, "The HMM regime detector identifies a NEUTRAL/SIDEWAYS macro environment.")

    return " ".join([ema_line, range_line, ret_line, regime_line])


def _momentum_paragraph(rsi: float, macd_diff: float, return_1: float) -> str:
    # RSI
    if rsi >= 75:
        rsi_line = (
            f"RSI stands at {rsi:.1f}, firmly in overbought territory — "
            "a short-term pullback or consolidation is a real possibility."
        )
    elif rsi >= 60:
        rsi_line = (
            f"RSI reads {rsi:.1f}, mildly elevated but not yet overbought, "
            "suggesting continued bullish momentum."
        )
    elif rsi <= 25:
        rsi_line = (
            f"RSI is at {rsi:.1f}, deeply oversold — historically this "
            "zone precedes a mean-reversion bounce."
        )
    elif rsi <= 40:
        rsi_line = (
            f"RSI at {rsi:.1f} is approaching oversold levels, "
            "indicating bearish short-term momentum."
        )
    else:
        rsi_line = (
            f"RSI is at a neutral {rsi:.1f}, reflecting balanced "
            "buying and selling pressure."
        )

    # MACD
    if macd_diff > 0:
        macd_line = (
            f"The MACD line is above its signal line (diff: {macd_diff:.4f}), "
            "a bullish crossover signal."
        )
    elif macd_diff < 0:
        macd_line = (
            f"The MACD line has crossed below its signal line (diff: {macd_diff:.4f}), "
            "a bearish momentum shift."
        )
    else:
        macd_line = "MACD is flat against its signal line — no clear directional bias."

    # Day return
    ret1_pct = return_1 * 100
    if abs(ret1_pct) < 0.2:
        ret_line = "Today's price change is minimal."
    elif ret1_pct > 0:
        ret_line = f"The stock moved up {ret1_pct:.2f}% today, adding to short-term strength."
    else:
        ret_line = f"Today's {abs(ret1_pct):.2f}% decline adds to near-term weakness."

    return " ".join([rsi_line, macd_line, ret_line])


def _volatility_paragraph(atr_pct: float, volatility_10: float) -> str:
    atr_pct_val = atr_pct * 100
    vol_pct     = volatility_10 * 100

    # ATR
    if atr_pct_val > 3.0:
        atr_line = (
            f"The Average True Range is {atr_pct_val:.2f}% of the stock price, "
            "indicating high daily swings — position sizing should be conservative."
        )
    elif atr_pct_val > 1.5:
        atr_line = (
            f"ATR is {atr_pct_val:.2f}% of price, reflecting moderate intraday volatility."
        )
    else:
        atr_line = (
            f"ATR is low at {atr_pct_val:.2f}% of price — the stock is moving "
            "within a tight range."
        )

    # Rolling vol
    if vol_pct > 2.5:
        vol_line = (
            f"10-day rolling return volatility is elevated at {vol_pct:.2f}%, "
            "suggesting the market is pricing in uncertainty."
        )
    elif vol_pct > 1.0:
        vol_line = (
            f"10-day volatility sits at {vol_pct:.2f}%, in the normal range "
            "for an actively traded stock."
        )
    else:
        vol_line = (
            f"10-day volatility is subdued at {vol_pct:.2f}%, consistent with "
            "a range-bound or low-conviction market."
        )

    return " ".join([atr_line, vol_line])


def _ai_models_paragraph(
    ml_prob: float,
    lstm_ret: float,
    tcn_ret: float,
    ppo_act: str,
) -> str:
    ml_pct = ml_prob * 100

    # ML
    if ml_pct >= 65:
        ml_line = (
            f"The ML ensemble assigns a {ml_pct:.1f}% probability to an upward move "
            "next week — a strongly bullish signal."
        )
    elif ml_pct >= 55:
        ml_line = (
            f"The ML ensemble gives a {ml_pct:.1f}% probability of a weekly gain — "
            "marginally bullish."
        )
    elif ml_pct <= 35:
        ml_line = (
            f"The ML ensemble puts the probability of an upward move at just {ml_pct:.1f}% — "
            "a bearish reading."
        )
    else:
        ml_line = (
            f"The ML ensemble returns a neutral {ml_pct:.1f}% probability — "
            "no strong directional edge."
        )

    # LSTM
    lstm_pct = lstm_ret * 100
    if abs(lstm_pct) < 0.1:
        lstm_line = "The LSTM network forecasts a near-flat 5-day return."
    elif lstm_pct > 0:
        lstm_line = (
            f"The LSTM deep-learning model forecasts a +{lstm_pct:.2f}% return "
            "over the next 5 trading days."
        )
    else:
        lstm_line = (
            f"The LSTM network forecasts a {lstm_pct:.2f}% decline "
            "over the next 5 trading days."
        )

    # TCN
    tcn_pct = tcn_ret * 100
    if tcn_pct > 0:
        tcn_line = (
            f"The Temporal CNN independently confirms the LSTM, predicting "
            f"+{tcn_pct:.2f}% upside."
        )
    elif tcn_pct < 0:
        tcn_line = (
            f"The Temporal CNN disagrees with a downside forecast of "
            f"{tcn_pct:.2f}%."
        )
    else:
        tcn_line = "The Temporal CNN returns a neutral forecast."

    # PPO
    ppo_phrases = {
        "BUY":  "The PPO reinforcement-learning agent has decided to BUY, having learned this maximises portfolio value in similar past conditions.",
        "SELL": "The PPO reinforcement-learning agent recommends SELL, based on reward signals from historical trading simulations.",
        "HOLD": "The PPO reinforcement-learning agent recommends HOLD, choosing not to enter a new position at this juncture.",
    }
    ppo_line = ppo_phrases.get(ppo_act, "The PPO agent returned an ambiguous signal.")

    return " ".join([ml_line, lstm_line, tcn_line, ppo_line])


# Human-readable labels for technical feature names
_FEATURE_LABELS: Dict[str, str] = {
    "rsi_norm":      "RSI (momentum oscillator)",
    "ema_spread":    "EMA crossover (trend alignment)",
    "macd_diff":     "MACD divergence (trend + momentum)",
    "atr_pct":       "ATR (daily price range / volatility)",
    "volatility_10": "10-day return volatility",
    "return_1":      "1-day price return",
    "return_5":      "5-day price return",
    "return_10":     "10-day price return",
    "range_position":"price position within 20-day range",
}


def _shap_paragraph(
    shap_ranked: List[Dict],
    feature_values: Dict[str, float],
) -> str:
    if not shap_ranked:
        return "No SHAP data available."

    lines = [
        "The three most influential features driving this prediction are:"
    ]

    for i, item in enumerate(shap_ranked[:3], 1):
        feat  = item["feature"]
        val   = item["shap"]
        dirn  = item["direction"]
        label = _FEATURE_LABELS.get(feat, feat.replace("_", " "))
        raw   = feature_values.get(feat)

        # Build value string
        if raw is not None:
            if feat == "rsi_norm":
                raw_str = f"current reading: RSI {raw*100:.1f}"
            elif feat in ("ema_spread", "atr_pct", "volatility_10",
                          "return_1", "return_5", "return_10"):
                raw_str = f"current reading: {raw*100:.2f}%"
            elif feat == "range_position":
                raw_str = f"current position: {raw*100:.0f}th percentile"
            else:
                raw_str = f"current value: {raw:.4f}"
        else:
            raw_str = ""

        arrow = "↑" if dirn == "bullish" else "↓"
        impact_str = f"SHAP impact: {val:+.4f}"

        lines.append(
            f"  {i}. **{label.title()}** — pushed the prediction {dirn} "
            f"({impact_str}) [{raw_str}] {arrow}"
        )

    # Remaining features
    if len(shap_ranked) > 3:
        others = shap_ranked[3:]
        bullish_others = [o["feature"] for o in others if o["direction"] == "bullish"]
        bearish_others = [o["feature"] for o in others if o["direction"] == "bearish"]

        support_lines = []
        if bullish_others:
            labels = [_FEATURE_LABELS.get(f, f.replace("_", " ")) for f in bullish_others]
            support_lines.append(f"Additional bullish support from: {', '.join(labels)}.")
        if bearish_others:
            labels = [_FEATURE_LABELS.get(f, f.replace("_", " ")) for f in bearish_others]
            support_lines.append(f"Bearish counterweights: {', '.join(labels)}.")
        lines.extend(support_lines)

    return "\n".join(lines)


def _news_paragraph(company: str, sentiment: float) -> str:
    if sentiment > 0.5:
        return (
            f"Recent news coverage of {company} is highly positive "
            f"(sentiment score: {sentiment:.2f}), which may accelerate bullish momentum."
        )
    elif sentiment > 0.2:
        return (
            f"News sentiment is mildly positive for {company} "
            f"(score: {sentiment:.2f}), adding a modest tailwind."
        )
    elif sentiment < -0.5:
        return (
            f"News flow around {company} is significantly negative "
            f"(score: {sentiment:.2f}) — headline risk is elevated."
        )
    elif sentiment < -0.2:
        return (
            f"Sentiment from recent news is slightly negative for {company} "
            f"(score: {sentiment:.2f}), acting as a mild headwind."
        )
    else:
        return (
            f"News sentiment is neutral for {company} "
            f"(score: {sentiment:.2f}), contributing no directional bias."
        )


def _summary_paragraph(
    company: str,
    action: str,
    conf_pct: float,
    score: float,
    regime: str,
) -> str:
    score_desc = (
        "strongly" if abs(score) >= 4
        else "moderately" if abs(score) >= 2.5
        else "marginally"
    )

    regime_risk = {
        "BULL": "with the macro wind behind it",
        "BEAR": "despite an unfavourable macro backdrop",
    }.get(regime, "in a mixed macro environment")

    action_map = {
        "BUY":  f"Sensei AI {score_desc} favours accumulating {company} {regime_risk}.",
        "SELL": f"Sensei AI {score_desc} recommends reducing exposure to {company} {regime_risk}.",
        "HOLD": f"Sensei AI recommends patience — signals for {company} are inconclusive {regime_risk}.",
    }

    first = action_map.get(action, f"Signals for {company} are mixed.")
    second = (
        f"The overall model confidence is {conf_pct:.1f}%. "
        "This analysis is for informational purposes only and does not constitute financial advice."
    )

    return f"{first} {second}"
