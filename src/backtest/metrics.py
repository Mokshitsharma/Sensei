import numpy as np


def calculate_metrics(equity_series):
    returns = equity_series.pct_change().dropna()

    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)

    cumulative = equity_series / equity_series.cummax()
    max_drawdown = cumulative.min() - 1

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }
