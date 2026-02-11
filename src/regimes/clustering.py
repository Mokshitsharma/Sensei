# src/regimes/clustering.py

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class MarketRegimeClustering:
    """
    Market regime detection using KMeans clustering.

    Regimes inferred from data:
        - BULL
        - BEAR
        - SIDEWAYS
    """

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        self.cluster_map: Dict[int, str] = {}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        data["return"] = data["close"].pct_change()
        data["volatility"] = data["return"].rolling(10).std()

        return data[["return", "volatility"]].dropna()

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit clustering model and infer regime labels.
        """
        features = self._prepare_features(df)

        X = self.scaler.fit_transform(features.values)
        clusters = self.model.fit_predict(X)

        features = features.assign(cluster=clusters)

        cluster_returns = (
            features.groupby("cluster")["return"].mean().to_dict()
        )

        sorted_clusters = sorted(
            cluster_returns.items(), key=lambda x: x[1]
        )

        self.cluster_map = {
            sorted_clusters[0][0]: "BEAR",
            sorted_clusters[1][0]: "SIDEWAYS",
            sorted_clusters[2][0]: "BULL",
        }

    def predict(self, df: pd.DataFrame) -> str:
        """
        Predict current market regime.
        """
        features = self._prepare_features(df)
        X = self.scaler.transform(features.values)
        cluster = self.model.predict(X)[-1]

        return self.cluster_map.get(cluster, "UNKNOWN")

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regime for each timestep.
        """
        features = self._prepare_features(df)
        X = self.scaler.transform(features.values)
        clusters = self.model.predict(X)

        regimes = [self.cluster_map[c] for c in clusters]
        index = df.index[-len(regimes):]

        return pd.Series(regimes, index=index, name="regime")