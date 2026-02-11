# src/domain/patterns.py

import pandas as pd
from typing import List, Dict


def detect_patterns(df: pd.DataFrame) -> List[Dict]:
    """
    Detect classical technical patterns.

    Returns a list of pattern events:
        {
            "date": pd.Timestamp,
            "type": str,
            "strength": float
        }
    """

    patterns: List[Dict] = []

    if len(df) < 50:
        return patterns

    data = df.copy()

    # -----------------------------
    # Golden / Death Cross
    # -----------------------------
    if "ema_20" in data.columns and "ema_50" in data.columns:
        prev = data.iloc[-2]
        curr = data.iloc[-1]

        if prev["ema_20"] < prev["ema_50"] and curr["ema_20"] > curr["ema_50"]:
            patterns.append(
                {
                    "date": curr["date"],
                    "type": "golden_cross",
                    "strength": 0.8,
                }
            )

        if prev["ema_20"] > prev["ema_50"] and curr["ema_20"] < curr["ema_50"]:
            patterns.append(
                {
                    "date": curr["date"],
                    "type": "death_cross",
                    "strength": 0.8,
                }
            )

    # -----------------------------
    # Breakout / Breakdown
    # -----------------------------
    lookback = 20
    recent = data.iloc[-lookback:]

    high_range = recent["high"].max()
    low_range = recent["low"].min()
    last_close = data.iloc[-1]["close"]

    if last_close > high_range:
        patterns.append(
            {
                "date": data.iloc[-1]["date"],
                "type": "breakout",
                "strength": 0.7,
            }
        )

    if last_close < low_range:
        patterns.append(
            {
                "date": data.iloc[-1]["date"],
                "type": "breakdown",
                "strength": 0.7,
            }
        )

    # -----------------------------
    # Simple Candlestick Patterns
    # -----------------------------
    candle = data.iloc[-1]
    body = abs(candle["close"] - candle["open"])
    range_ = candle["high"] - candle["low"]

    if range_ > 0:
        body_ratio = body / range_

        # Doji
        if body_ratio < 0.1:
            patterns.append(
                {
                    "date": candle["date"],
                    "type": "doji",
                    "strength": 0.4,
                }
            )

        # Strong bullish candle
        if candle["close"] > candle["open"] and body_ratio > 0.7:
            patterns.append(
                {
                    "date": candle["date"],
                    "type": "bullish_engulfing",
                    "strength": 0.6,
                }
            )

        # Strong bearish candle
        if candle["close"] < candle["open"] and body_ratio > 0.7:
            patterns.append(
                {
                    "date": candle["date"],
                    "type": "bearish_engulfing",
                    "strength": 0.6,
                }
            )

    return patterns