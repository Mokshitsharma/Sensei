# src/regimes/hmm.py

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class MarketRegimeHMM:
    """
    Hidden Markov Model for detecting market regimes.
    """

    def __init__(self, n_components=2):
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",   # ← IMPORTANT FIX
            n_iter=200,
            random_state=42,
        )
        self.fitted = False

    # =====================================================
    # FIT MODEL
    # =====================================================
    def fit(self, df: pd.DataFrame):

        if df is None or len(df) < 30:
            raise ValueError("Not enough data to fit HMM")

        # ------------------------------------------
        # Use returns ONLY (stationary series)
        # ------------------------------------------
        if "close" in df.columns:
            returns = df["close"].pct_change()
        elif "Close" in df.columns:
            returns = df["Close"].pct_change()
        else:
            raise ValueError("No close column found for HMM")

        returns = returns.dropna()

        # Remove extreme outliers (stability)
        returns = returns[abs(returns) < 0.5]

        if len(returns) < 20:
            raise ValueError("Insufficient clean return samples for HMM")

        X = returns.values.reshape(-1, 1)

        # Add tiny noise to prevent singular covariance
        X = X + 1e-6 * np.random.randn(*X.shape)

        self.model.fit(X)
        self.fitted = True

    # =====================================================
    # PREDICT REGIME
    # =====================================================
    def predict(self, df: pd.DataFrame):

        if not self.fitted:
            raise ValueError("HMM model not fitted")

        if "close" in df.columns:
            returns = df["close"].pct_change()
        elif "Close" in df.columns:
            returns = df["Close"].pct_change()
        else:
            raise ValueError("No close column found for HMM")

        returns = returns.dropna()
        returns = returns[abs(returns) < 0.5]

        X = returns.values.reshape(-1, 1)

        states = self.model.predict(X)

        # Assume higher mean return state = BULL
        means = self.model.means_.flatten()

        bull_state = np.argmax(means)
        current_state = states[-1]

        if current_state == bull_state:
            return "BULL"
        else:
            return "BEAR"
