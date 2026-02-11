# src/rl/train.py

import pandas as pd
from pathlib import Path

from src.data.prices import load_prices
from src.domain.indicators import add_indicators
from src.ml.features import build_features
from src.rl.env import TradingEnv
from src.rl.agent import PPOTradingAgent


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train_rl_agent(
    ticker: str,
    timeframe: str = "2y",
    timesteps: int = 200_000,
    model_name: str = "ppo_agent",
) -> None:
    """
    Train PPO trading agent on historical data.
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
    # Environment
    # -----------------------------
    env = TradingEnv(
        df=df,
        feature_cols=feature_cols,
        initial_balance=100_000.0,
        transaction_cost=0.001,
    )

    # -----------------------------
    # PPO Agent
    # -----------------------------
    agent = PPOTradingAgent(env)

    # -----------------------------
    # Training
    # -----------------------------
    agent.train(timesteps=timesteps)

    # -----------------------------
    # Save model
    # -----------------------------
    save_path = MODEL_DIR / model_name
    agent.save(str(save_path))

    print(f"PPO agent saved to {save_path}")

    # -----------------------------
    # Quick evaluation run
    # -----------------------------
    obs = env.reset()
    done = False

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

    print(
        f"Final Net Worth: {info['net_worth']:.2f} | "
        f"Balance: {info['balance']:.2f}"
    )


if __name__ == "__main__":
    # Example usage
    train_rl_agent(
        ticker="HDFC Bank",
        timeframe="2y",
        timesteps=300_000,
        model_name="ppo_hdfc",
    )
