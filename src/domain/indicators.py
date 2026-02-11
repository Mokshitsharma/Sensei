# src/domain/indicators.py

import pandas as pd
import numpy as np

def _ensure_flat_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).lower() for c in df.columns]
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add core technical indicators to OHLCV dataframe.

    Required columns:
        date, open, high, low, close, volume
    """

    data = df.copy()

    # -----------------------------
    # EMA (trend)
    # -----------------------------
    data["ema_20"] = data["close"].ewm(span=20, adjust=False).mean()
    data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()

    # -----------------------------
    # RSI (momentum)
    # -----------------------------
    delta = data["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    data["rsi"] = 100 - (100 / (1 + rs))

    # -----------------------------
    # MACD (trend + momentum)
    # -----------------------------
    ema_12 = data["close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["close"].ewm(span=26, adjust=False).mean()

    data["macd"] = ema_12 - ema_26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    # -----------------------------
    # ATR (volatility)
    # -----------------------------
    high_low = data["high"] - data["low"]
    high_close = (data["high"] - data["close"].shift()).abs()
    low_close = (data["low"] - data["close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["atr"] = tr.rolling(14).mean()

    return data