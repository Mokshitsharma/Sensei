# src/dl/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List


class TimeSeriesDataset(Dataset):
    """
    Rolling-window dataset for time-series forecasting.

    Each sample:
        X -> (seq_len, num_features)
        y -> (1,)

    Used by:
        - LSTM
        - Temporal CNN
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seq_len: int = 30,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len

        self._validate()

    def _validate(self) -> None:
        if len(self.df) <= self.seq_len:
            raise ValueError("DataFrame too small for given sequence length")

        missing = set(self.feature_cols + [self.target_col]) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.df) - self.seq_len

    def __getitem__(self, idx: int):
        start = idx
        end = idx + self.seq_len

        x = self.df.loc[start:end - 1, self.feature_cols].values
        y = self.df.loc[end, self.target_col]

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor([y], dtype=torch.float32)

        return x_tensor, y_tensor