# src/rl/env.py

import gym
import numpy as np
import pandas as pd
from gym import spaces
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

    def reset(self):
        self._reset_state()
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        obs = self.df.loc[self.current_step, self.feature_cols].values
        return obs.astype(np.float32)

    def step(self, action: int):
        done = False
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
            done = True

        obs = self._get_observation() if not done else None

        info = {
            "net_worth": self.net_worth,
            "balance": self.balance,
            "position": self.position,
        }

        return obs, reward, done, info

    def render(self, mode="human") -> None:
        print(
            f"Step: {self.current_step} | "
            f"Net Worth: {self.net_worth:.2f} | "
            f"Position: {self.position:.4f}"
        )