# src/domain/backtest.py

import pandas as pd
import numpy as np
from typing import Callable, Dict


class Backtester:
    """
    Event-driven backtester for BUY / SELL / HOLD strategies.

    Designed for:
        - ML / DL / RL signals
        - regime-aware strategies
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        position_size: float = 1.0,
        transaction_cost: float = 0.0005,
    ) -> None:
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost

    def run(
        self,
        df: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame, int], Dict[str, object]],
    ) -> Dict[str, object]:
        """
        Run backtest.

        signal_fn(df, i) must return:
            {
                "action": "BUY" | "SELL" | "HOLD"
            }
        """

        cash = self.initial_capital
        position = 0.0
        equity_curve = []

        for i in range(len(df)):
            price = df.loc[i, "close"]
            signal = signal_fn(df, i)
            action = signal["action"]

            if action == "BUY" and cash > 0:
                units = (cash * self.position_size) / price
                cost = units * price * self.transaction_cost
                cash -= units * price + cost
                position += units

            elif action == "SELL" and position > 0:
                proceeds = position * price
                cost = proceeds * self.transaction_cost
                cash += proceeds - cost
                position = 0.0

            equity = cash + position * price
            equity_curve.append(equity)

        equity_series = pd.Series(equity_curve, index=df.index)

        returns = equity_series.pct_change().dropna()

        return {
            "equity_curve": equity_series,
            "total_return": (equity_series.iloc[-1] / self.initial_capital) - 1,
            "cagr": self._cagr(equity_series),
            "max_drawdown": self._max_drawdown(equity_series),
            "sharpe": self._sharpe_ratio(returns),
        }

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return drawdown.min()

    @staticmethod
    def _sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        if returns.std() == 0:
            return 0.0
        return np.sqrt(252) * (returns.mean() - risk_free_rate) / returns.std()

    @staticmethod
    def _cagr(equity: pd.Series) -> float:
        years = len(equity) / 252
        if years == 0:
            return 0.0
        return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1