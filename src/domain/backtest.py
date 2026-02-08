# src/domain/backtest.py
import pandas as pd
from typing import Dict, List

from src.domain.signals import generate_signal


def backtest_strategy(
    df: pd.DataFrame,
    fundamentals: Dict[str, float],
) -> Dict[str, object]:
    equity = 100000.0  # starting capital
    peak = equity
    drawdowns = []
    equity_curve = []

    position = None
    entry_price = 0.0
    trades: List[float] = []

    for i in range(50, len(df)):
        window = df.iloc[:i]
        latest = window.iloc[-1]

        signal_data = generate_signal(
            indicators={
                "rsi": latest["rsi"],
                "macd": latest["macd"],
                "macd_signal": latest["macd_signal"],
            },
            patterns=[],
            fundamentals=fundamentals,
        )

        signal = signal_data["signal"]

        if signal == "BUY" and position is None:
            position = "LONG"
            entry_price = latest["close"]

        elif signal == "SELL" and position == "LONG":
            exit_price = latest["close"]
            ret = (exit_price - entry_price) / entry_price
            equity *= (1 + ret)
            trades.append(ret)
            position = None

        peak = max(peak, equity)
        drawdown = (peak - equity) / peak
        drawdowns.append(drawdown)
        equity_curve.append(equity)

    if not trades:
        return {
            "trades": 0,
            "win_rate": 0,
            "return_pct": 0,
            "max_drawdown": 0,
            "equity_curve": [],
        }

    wins = [t for t in trades if t > 0]

    return {
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 2),
        "return_pct": round((equity - 100000) / 100000 * 100, 2),
        "max_drawdown": round(max(drawdowns) * 100, 2),
        "equity_curve": equity_curve,
    }
