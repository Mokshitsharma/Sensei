# src/rl/evaluate.py

import numpy as np
import pandas as pd
from typing import Dict

from src.rl.env import TradingEnv
from src.rl.agent import PPOTradingAgent
from src.data.prices import load_prices
from src.domain.indicators import add_indicators
from src.ml.features import build_features


def _sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    if returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate
    return np.sqrt(252) * excess.mean() / excess.std()


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    return float(drawdown.max())


def evaluate_agent(
    ticker: str,
    model_path: str,
    timeframe: str = "2y",
) -> Dict[str, float]:
    """
    Evaluate trained PPO agent on historical data.
    """

    # -----------------------------
    # Load & prepare data
    # -----------------------------
    df = load_prices(ticker, timeframe)
    df = add_indicators(df)
    df = build_features(df)

    feature_cols = [
        "rsi_norm",
        "ema_spread",
        "macd_diff",
        "atr_pct",
    ]

    df = df.dropna().reset_index(drop=True)

    # -----------------------------
    # Environment & Agent
    # -----------------------------
    env = TradingEnv(df=df, feature_cols=feature_cols)
    agent = PPOTradingAgent(env, model_path=model_path)

    # -----------------------------
    # Run evaluation
    # -----------------------------
    obs = env.reset()
    done = False

    equity_curve = []
    rewards = []

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        equity_curve.append(info["net_worth"])
        rewards.append(reward)

    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    # -----------------------------
    # Metrics
    # -----------------------------
    total_return = (equity[-1] - equity[0]) / equity[0]
    sharpe = _sharpe_ratio(returns)
    max_dd = _max_drawdown(equity)
    win_rate = (returns > 0).mean()

    results = {
        "Total Return (%)": round(total_return * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Win Rate (%)": round(win_rate * 100, 2),
        "Final Net Worth": round(equity[-1], 2),
    }

    for k, v in results.items():
        print(f"{k}: {v}")

    return results


if __name__ == "__main__":
    evaluate_agent(
        ticker="HDFC Bank",
        model_path="models/ppo_hdfc",
        timeframe="2y",
    )
