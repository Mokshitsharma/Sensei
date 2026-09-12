# src/rl/env.py

import random

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import List


class TradingEnv(gym.Env):
    """
    Reinforcement Learning trading environment.

    Actions:
        0 = HOLD
        1 = BUY
        2 = SELL

    Observation:
        Technical + ML features at time t

    Reward:
        Change in portfolio value
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        initial_balance: float = 100_000.0,
        transaction_cost: float = 0.001,
    ) -> None:
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(feature_cols),),
            dtype=np.float32,
        )

        self._reset_state()

    def _reset_state(self) -> None:
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0          # number of shares held
        self.entry_price = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        obs = self.df.loc[self.current_step, self.feature_cols].values
        return obs.astype(np.float32)

    def step(self, action: int):
        terminated = False
        truncated = False
        price = self.df.loc[self.current_step, "close"]

        prev_net_worth = self.net_worth

        # -----------------------
        # Execute action
        # -----------------------
        if action == 1 and self.position == 0:  # BUY
            self.position = self.balance / price
            cost = self.balance * self.transaction_cost
            self.balance -= cost
            self.entry_price = price

        elif action == 2 and self.position > 0:  # SELL
            self.balance = self.position * price
            cost = self.balance * self.transaction_cost
            self.balance -= cost
            self.position = 0
            self.entry_price = 0.0

        # -----------------------
        # Update net worth
        # -----------------------
        self.net_worth = self.balance + self.position * price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        reward = self.net_worth - prev_net_worth

        # Penalize drawdown
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        reward -= drawdown * 0.1

        # -----------------------
        # Next step
        # -----------------------
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            terminated = True

        # Always a real observation (never None) — current_step is still a
        # valid row index at termination, and SB3's VecEnv stores this
        # exact return as `terminal_observation` for value bootstrapping.
        obs = self._get_observation()

        info = {
            "net_worth": self.net_worth,
            "balance": self.balance,
            "position": self.position,
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode="human") -> None:
        print(
            f"Step: {self.current_step} | "
            f"Net Worth: {self.net_worth:.2f} | "
            f"Position: {self.position:.4f}"
        )


class MultiTickerTradingEnv(gym.Env):
    """Wraps one TradingEnv per ticker and picks a random one on each
    reset(), so a single PPO agent trains across every stock's price
    history instead of just one. Each episode still runs entirely within
    one ticker's data — only episode *selection* is pooled, never a single
    episode's steps mixing tickers."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        dfs: List[pd.DataFrame],
        feature_cols: List[str],
        initial_balance: float = 100_000.0,
        transaction_cost: float = 0.001,
    ) -> None:
        super().__init__()
        if not dfs:
            raise ValueError("MultiTickerTradingEnv needs at least one ticker's data")

        self.envs = [
            TradingEnv(
                df=df,
                feature_cols=feature_cols,
                initial_balance=initial_balance,
                transaction_cost=transaction_cost,
            )
            for df in dfs
        ]
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.active: TradingEnv = self.envs[0]

    def reset(self, seed=None, options=None):
        self.active = random.choice(self.envs)
        return self.active.reset(seed=seed, options=options)

    def step(self, action: int):
        return self.active.step(action)

    def render(self, mode="human") -> None:
        self.active.render(mode)
