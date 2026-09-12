# src/rl/agent.py

import functools

from src.utils import numpy_compat
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional
import gymnasium as gym


@functools.lru_cache(maxsize=None)
def _load_ppo_cached(model_path: str) -> PPO:
    """Cached per model_path — every /analysis request was re-reading and
    unpickling the PPO zip from disk. Loaded without a bound env: `.act()`
    only calls `model.predict()`, which doesn't need one, so a single
    inference-only instance is safe to share across requests/tickers."""
    numpy_compat.patch()
    return PPO.load(model_path, env=None)


class PPOTradingAgent:
    """
    PPO agent wrapper for trading environment.

    Action space:
        0 = HOLD
        1 = BUY
        2 = SELL
    """

    def __init__(
        self,
        env: gym.Env,
        model_path: Optional[str] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 2048,
        batch_size: int = 64,
    ) -> None:
        self.env = DummyVecEnv([lambda: env])

        if model_path:
            self.model = _load_ppo_cached(model_path)
        else:
            self.model = PPO(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=learning_rate,
                gamma=gamma,
                n_steps=n_steps,
                batch_size=batch_size,
                verbose=1,
            )

    def train(self, timesteps: int = 100_000) -> None:
        """
        Train PPO agent.
        """
        self.model.learn(total_timesteps=timesteps)

    def save(self, path: str) -> None:
        """
        Save trained agent.
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """
        Load trained agent.
        """
        numpy_compat.patch()
        self.model = PPO.load(path, env=self.env)

    def act(self, observation):
        """
        Predict action given observation.
        """
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
