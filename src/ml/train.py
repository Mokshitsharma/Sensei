import os
import joblib

from src.data.prices import load_prices
from src.ml.features import build_features
from src.ml.model import train_model


def train_ml_model(
    ticker="HDFCBANK.NS",
    timeframe="1y",
    model_name="ml_return_model",
):
    print("Loading price data...")
    df = load_prices(ticker, timeframe)

    print("Building features...")
    df = build_features(df)

    print("Training ML model...")
    model = train_model(df)

    os.makedirs("models", exist_ok=True)

    model_path = f"models/{model_name}.joblib"
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_ml_model()