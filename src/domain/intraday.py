# src/domain/intraday.py

import pandas as pd
import numpy as np


def add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add intraday-specific features.

    Required columns:
        date, open, high, low, close, volume
    """

    data = df.copy()

    # -----------------------------
    # VWAP (intraday anchor)
    # -----------------------------
    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    cumulative_vp = (typical_price * data["volume"]).cumsum()
    cumulative_vol = data["volume"].cumsum()

    data["vwap"] = cumulative_vp / (cumulative_vol + 1e-9)

    # -----------------------------
    # Fast EMAs (intraday trend)
    # -----------------------------
    data["ema_9"] = data["close"].ewm(span=9, adjust=False).mean()
    data["ema_21"] = data["close"].ewm(span=21, adjust=False).mean()

    # -----------------------------
    # VWAP distance (%)
    # -----------------------------
    data["vwap_dist_pct"] = (data["close"] - data["vwap"]) / data["vwap"]

    # -----------------------------
    # Intraday momentum
    # -----------------------------
    data["return_1"] = data["close"].pct_change()
    data["return_5"] = data["close"].pct_change(5)

    # -----------------------------
    # Session flags (India market)
    # -----------------------------
    data["hour"] = data["date"].dt.hour
    data["is_opening"] = data["hour"].between(9, 10)
    data["is_midday"] = data["hour"].between(11, 13)
    data["is_closing"] = data["hour"].between(14, 15)

    return data


def intraday_bias(df: pd.DataFrame) -> str:
    """
    Determine intraday directional bias.

    Returns:
        BULLISH | BEARISH | NEUTRAL
    """

    latest = df.iloc[-1]

    if (
        latest["close"] > latest["vwap"]
        and latest["ema_9"] > latest["ema_21"]
    ):
        return "BULLISH"

    if (
        latest["close"] < latest["vwap"]
        and latest["ema_9"] < latest["ema_21"]
    ):
        return "BEARISH"

    return "NEUTRAL"