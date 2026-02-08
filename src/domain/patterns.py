# src/domain/patterns.py
import pandas as pd
from typing import List, Dict


def detect_patterns(
    df: pd.DataFrame,
    lookback: int = 60,
) -> List[Dict]:
    """
    Detects recent technical patterns and returns event markers.

    Returns:
    [
        {
            "type": "golden_cross" | "death_cross" | "breakout",
            "date": pd.Timestamp,
            "price": float,
        }
    ]
    """

    events = []

    if len(df) < lookback:
        return events

    recent = df.tail(lookback).reset_index(drop=True)

    # EMA crosses
    for i in range(1, len(recent)):
        prev = recent.iloc[i - 1]
        curr = recent.iloc[i]

        # Golden Cross
        if prev["ema_20"] < prev["ema_50"] and curr["ema_20"] > curr["ema_50"]:
            events.append(
                {
                    "type": "golden_cross",
                    "date": curr["date"],
                    "price": curr["close"],
                }
            )

        # Death Cross
        if prev["ema_20"] > prev["ema_50"] and curr["ema_20"] < curr["ema_50"]:
            events.append(
                {
                    "type": "death_cross",
                    "date": curr["date"],
                    "price": curr["close"],
                }
            )

    # Breakouts (20-day high)
    rolling_high = recent["high"].rolling(20).max()

    for i in range(20, len(recent)):
        if recent.iloc[i]["close"] > rolling_high.iloc[i - 1]:
            events.append(
                {
                    "type": "breakout",
                    "date": recent.iloc[i]["date"],
                    "price": recent.iloc[i]["close"],
                }
            )

    return events
