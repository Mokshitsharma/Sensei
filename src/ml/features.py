# src/ml/features.py

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML features from price + indicators.

    Output includes:
        - normalized technical features
        - volatility & momentum features
        - future return targets (for supervised ML)

    NO leakage. NO signals.
    """

    data = df.copy()

    # -----------------------------
    # Normalized Technicals
    # -----------------------------
    data["rsi_norm"] = data["rsi"] / 100.0

    data["ema_spread"] = (
        data["ema_20"] - data["ema_50"]
    ) / data["close"]

    data["macd_diff"] = (
        data["macd"] - data["macd_signal"]
    )

    # -----------------------------
    # Volatility Features
    # -----------------------------
    data["atr_pct"] = data["atr"] / data["close"]

    data["volatility_10"] = (
        data["close"].pct_change().rolling(10).std()
    )

    # -----------------------------
    # Momentum Features
    # -----------------------------
    data["return_1"] = data["close"].pct_change()
    data["return_5"] = data["close"].pct_change(5)
    data["return_10"] = data["close"].pct_change(10)

    # -----------------------------
    # Price Positioning
    # -----------------------------
    rolling_high = data["high"].rolling(20).max()
    rolling_low = data["low"].rolling(20).min()

    data["range_position"] = (
        (data["close"] - rolling_low)
        / (rolling_high - rolling_low + 1e-9)
    )

    # -----------------------------
    # Supervised Targets (future)
    # -----------------------------
    data["future_return_5d"] = (
        data["close"].shift(-5) / data["close"] - 1
    )

    data["future_direction_5d"] = (
        data["future_return_5d"] > 0
    ).astype(int)

    # -----------------------------
    # Cleanup
    # -----------------------------
    data = data.replace([np.inf, -np.inf], np.nan)

    return data