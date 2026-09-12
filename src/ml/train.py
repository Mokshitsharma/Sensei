# src/ml/train.py

import os
from typing import Iterable, Optional

import joblib
import pandas as pd

from src.data.nifty50 import NIFTY_50
from src.ml.model import build_model, train_model
from src.training.pooled_data import build_pooled_dataset
from src.utils.config import FEATURE_COLUMNS, TARGET_RETURN


def train_ml_model_pooled(
    tickers: Optional[Iterable[str]] = None,
    timeframe: str = "2y",
    model_name: str = "ml_return_model",
) -> None:
    """Trains one Random Forest on data pooled across every given ticker
    (defaults to all NIFTY 50), on the canonical 9-feature schema."""
    tickers = list(tickers or NIFTY_50.values())

    print(f"Building pooled dataset from {len(tickers)} tickers...")
    pairs = build_pooled_dataset(tickers, timeframe=timeframe)
    if not pairs:
        raise RuntimeError("No usable ticker data — check network/yfinance access")

    pooled = pd.concat([df for _, df in pairs], ignore_index=True)
    print(f"Pooled {len(pooled)} rows from {len(pairs)}/{len(tickers)} tickers")

    X = pooled[FEATURE_COLUMNS]
    y = pooled[TARGET_RETURN]

    model = build_model("random_forest", task="regression")
    model = train_model(model, X, y)

    os.makedirs("models", exist_ok=True)
    path = f"models/{model_name}.joblib"
    joblib.dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    train_ml_model_pooled()
