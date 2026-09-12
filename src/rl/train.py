# src/rl/train.py

from typing import Iterable, Optional
from pathlib import Path

from src.data.nifty50 import NIFTY_50
from src.training.pooled_data import build_pooled_dataset
from src.rl.env import MultiTickerTradingEnv
from src.rl.agent import PPOTradingAgent


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = ["rsi_norm", "ema_spread", "macd_diff", "atr_pct"]


def train_rl_agent_pooled(
    tickers: Optional[Iterable[str]] = None,
    timeframe: str = "2y",
    timesteps: int = 200_000,
    model_name: str = "ppo_general",
) -> None:
    """Trains one PPO agent across every given ticker (defaults to all
    NIFTY 50) via MultiTickerTradingEnv, which picks a random ticker's
    price history for each episode."""
    tickers = list(tickers or NIFTY_50.values())

    print(f"Building pooled dataset from {len(tickers)} tickers...")
    pairs = build_pooled_dataset(tickers, timeframe=timeframe, min_rows=60)
    if not pairs:
        raise RuntimeError("No usable ticker data — check network/yfinance access")

    dfs = [df for _, df in pairs]
    print(f"Training PPO across {len(dfs)}/{len(tickers)} tickers")

    env = MultiTickerTradingEnv(
        dfs=dfs,
        feature_cols=FEATURE_COLS,
        initial_balance=100_000.0,
        transaction_cost=0.001,
    )

    agent = PPOTradingAgent(env)
    agent.train(timesteps=timesteps)

    save_path = MODEL_DIR / model_name
    agent.save(str(save_path))
    print(f"PPO agent saved to {save_path}")

    # Quick evaluation run on whichever ticker gets picked
    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print(
        f"Final Net Worth: {info['net_worth']:.2f} | "
        f"Balance: {info['balance']:.2f}"
    )


if __name__ == "__main__":
    train_rl_agent_pooled(
        timeframe="2y",
        timesteps=200_000,
        model_name="ppo_general",
    )
