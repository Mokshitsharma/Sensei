# src/regimes/hmm.py

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from typing import Dict


class MarketRegimeHMM:
    """
    Hidden Markov Model for market regime detection.

    Regimes (learned, not hard-coded):
        - Bull
        - Bear
        - Sideways
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        self.state_map: Dict[int, str] = {}

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Features:
            - log return
            - rolling volatility
        """
        data = df.copy()

        data["log_return"] = np.log(data["close"] / data["close"].shift(1))
        data["volatility"] = data["log_return"].rolling(10).std()

        features = data[["log_return", "volatility"]].dropna()
        return features.values

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit HMM and infer regime labels.
        """
        X = self._prepare_features(df)
        self.model.fit(X)

        hidden_states = self.model.predict(X)

        regime_stats = {}
        for state in range(self.n_states):
            returns = X[hidden_states == state, 0]
            regime_stats[state] = returns.mean()

        sorted_states = sorted(
            regime_stats.items(), key=lambda x: x[1]
        )

        self.state_map = {
            sorted_states[0][0]: "BEAR",
            sorted_states[1][0]: "SIDEWAYS",
            sorted_states[2][0]: "BULL",
        }

    def predict(self, df: pd.DataFrame) -> str:
        """
        Predict current market regime.
        """
        X = self._prepare_features(df)
        state = self.model.predict(X)[-1]
        return self.state_map.get(state, "UNKNOWN")

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regime for each timestep.
        """
        X = self._prepare_features(df)
        states = self.model.predict(X)
        regimes = [self.state_map[s] for s in states]

        index = df.index[-len(regimes):]
        return pd.Series(regimes, index=index, name="regime")