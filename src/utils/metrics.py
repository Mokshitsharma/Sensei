# src/utils/metrics.py

import numpy as np
import pandas as pd
from typing import Dict


# =============================
# Return / Portfolio Metrics
# =============================

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods: int = 252,
) -> float:
    """
    Annualized Sharpe Ratio.
    """
    if returns.std() == 0:
        return 0.0

    excess = returns - risk_free_rate / periods
    return float(np.sqrt(periods) * excess.mean() / excess.std())


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods: int = 252,
) -> float:
    """
    Annualized Sortino Ratio (downside risk).
    """
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0.0

    excess = returns - risk_free_rate / periods
    return float(np.sqrt(periods) * excess.mean() / downside.std())


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum drawdown.
    """
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def cagr(
    equity: pd.Series,
    periods: int = 252,
) -> float:
    """
    Compound Annual Growth Rate.
    """
    if len(equity) < 2:
        return 0.0

    years = len(equity) / periods
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def portfolio_metrics(
    equity: pd.Series,
) -> Dict[str, float]:
    """
    Compute core portfolio performance metrics.
    """
    returns = equity.pct_change().dropna()

    return {
        "cagr": cagr(equity),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
    }


# =============================
# Trade-Level Metrics
# =============================

def trade_metrics(trade_returns: pd.Series) -> Dict[str, float]:
    """
    Metrics based on per-trade returns.
    """
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0.0

    profit_factor = (
        wins.sum() / abs(losses.sum())
        if abs(losses.sum()) > 0
        else np.inf
    )

    return {
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "num_trades": int(len(trade_returns)),
    }


# =============================
# Prediction Metrics
# =============================

def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Directional accuracy for price movement.
    """
    true_dir = (y_true > 0).astype(int)
    pred_dir = (y_pred > 0).astype(int)
    return float((true_dir == pred_dir).mean())


def regression_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Regression error metrics (numpy-only).
    """
    error = y_pred - y_true

    return {
        "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
    }