# src/backtest/engine.py

import numpy as np
import pandas as pd


def run_backtest(price_df, signals_series, initial_capital=100000):
    """
    Simple long-only backtest using BUY / SELL signals.
    """

    df = price_df.copy()

    # ------------------------------------------
    # Handle column naming safely
    # ------------------------------------------
    if "close" in df.columns:
        close_col = "close"
    elif "Close" in df.columns:
        close_col = "Close"
    else:
        raise ValueError("No close column found in price data")

    df["signal"] = signals_series.values

    cash = initial_capital
    position = 0
    equity_curve = []

    for i in range(len(df)):
        price = df[close_col].iloc[i]
        signal = df["signal"].iloc[i]

        # BUY
        if signal == "BUY" and cash > 0:
            position = cash / price
            cash = 0

        # SELL
        elif signal == "SELL" and position > 0:
            cash = position * price
            position = 0

        net_worth = cash + position * price
        equity_curve.append(net_worth)

    df["equity"] = equity_curve

    return df
