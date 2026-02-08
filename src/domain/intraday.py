# src/domain/intraday.py
from typing import Dict


def intraday_scalping_signal(latest: dict) -> Dict[str, float | str]:
    """
    Intraday scalping signal using EMA + RSI + ATR.
    """

    entry = latest["close"]
    atr = latest["atr"]

    if atr is None:
        return {"signal": "NO TRADE"}

    # LONG
    if (
        latest["ema_20"] > latest["ema_50"]
        and 40 <= latest["rsi"] <= 60
    ):
        return {
            "signal": "BUY",
            "entry": round(entry, 2),
            "stop_loss": round(entry - atr, 2),
            "target": round(entry + (1.5 * atr), 2),
        }

    # SHORT
    if (
        latest["ema_20"] < latest["ema_50"]
        and 40 <= latest["rsi"] <= 60
    ):
        return {
            "signal": "SELL",
            "entry": round(entry, 2),
            "stop_loss": round(entry + atr, 2),
            "target": round(entry - (1.5 * atr), 2),
        }

    return {"signal": "NO TRADE"}
