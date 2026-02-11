# src/ml/evaluation.py

from typing import Dict
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate regression model (e.g. future returns).

    Returns standard regression metrics.
    """

    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_directional(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate directional accuracy (up/down).

    Converts values to sign-based classification.
    """

    true_dir = (y_true > 0).astype(int)
    pred_dir = (y_pred > 0).astype(int)

    return {
        "accuracy": float(accuracy_score(true_dir, pred_dir)),
        "precision": float(precision_score(true_dir, pred_dir, zero_division=0)),
        "recall": float(recall_score(true_dir, pred_dir, zero_division=0)),
        "f1": float(f1_score(true_dir, pred_dir, zero_division=0)),
    }


def evaluate_full_ml(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Full ML evaluation:
        - regression quality
        - directional correctness
    """

    return {
        "regression": evaluate_regression(y_true, y_pred),
        "directional": evaluate_directional(y_true, y_pred),
    }