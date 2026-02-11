# src/pipeline/signal_pipeline.py

import torch
import pandas as pd
from typing import Dict

from src.domain.indicators import add_indicators
from src.domain.signals import generate_signal

from src.ml.features import build_features
from src.ml.predict import predict_next_week

from src.dl.lstm import load_model as load_lstm
from src.dl.temporal_cnn import load_model as load_tcn

from src.regimes.hmm import MarketRegimeHMM
from src.rl.agent import PPOTradingAgent
from src.rl.env import TradingEnv


def run_signal_pipeline(
    price_df: pd.DataFrame,
    fundamentals: Dict[str, float],
    company: str,
    lstm_model_path: str,
    tcn_model_path: str,
    ppo_model_path: str,
) -> Dict[str, object]:
    """
    Full inference pipeline.

    Returns all intermediate signals required by decision_engine.
    """

    # -----------------------------
    # Indicators
    # -----------------------------
    df = add_indicators(price_df)
    latest = df.iloc[-1]

    # -----------------------------
    # Rule-based signal
    # -----------------------------
    rule_signal = generate_signal(
        indicators={
            "rsi": latest["rsi"],
            "macd": latest["macd"],
            "macd_signal": latest["macd_signal"],
        },
        patterns=[],
        fundamentals=fundamentals,
    )

    # -----------------------------
    # ML features
    # -----------------------------
    feature_df = build_features(df)
    latest_features = feature_df.iloc[-1:]

    ml_out = predict_next_week(
        df=feature_df,
        company=company,
    )

    ml_prob_up = ml_out["confidence"] / 100

    # -----------------------------
    # Deep Learning (LSTM + TCN)
    # -----------------------------
    feature_cols = [
        "rsi_norm",
        "ema_spread",
        "macd_diff",
        "atr_pct",
    ]

    seq_len = 30
    seq = feature_df[feature_cols].tail(seq_len).values
    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

    lstm = load_lstm(lstm_model_path, num_features=len(feature_cols))
    tcn = load_tcn(tcn_model_path, num_features=len(feature_cols))

    lstm_return = float(lstm(seq_tensor).item())
    tcn_return = float(tcn(seq_tensor).item())

    # -----------------------------
    # Regime Detection
    # -----------------------------
    hmm = MarketRegimeHMM()
    hmm.fit(df)
    regime = hmm.predict(df)

    # -----------------------------
    # Reinforcement Learning (PPO)
    # -----------------------------
    env = TradingEnv(
        df=feature_df,
        feature_cols=feature_cols,
    )

    agent = PPOTradingAgent(env, model_path=ppo_model_path)
    obs = env.reset()
    ppo_action = agent.act(obs)

    return {
        "rule_signal": rule_signal,
        "ml_prob_up": ml_prob_up,
        "lstm_return": lstm_return,
        "tcn_return": tcn_return,
        "regime": regime,
        "ppo_action": ppo_action,
    }