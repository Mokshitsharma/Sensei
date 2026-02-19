# src/ml/predict.py

from typing import Dict
import numpy as np
import pandas as pd

from src.ml.model import load_model, predict, predict_proba


def predict_next_week(
    df: pd.DataFrame,
    company: str,
    model_name: str = "ml_return_model",
    task: str = "regression",
) -> Dict[str, float]:
    """
    Predict next-week price movement using classical ML.
    If model not found, return neutral prediction.
    """

    try:
        model = load_model(model_name)
    except Exception:
        return {
            "prediction": 0.0,
            "direction": "NEUTRAL",
            "confidence": 0.0,
        }

    feature_cols = [
        c for c in df.columns
        if c not in (
            "future_return_5d",
            "future_direction_5d",
            "date",
        )
    ]

    latest_X = df[feature_cols].iloc[-1:].values
    y_pred = predict(model, latest_X)[0]

    if task == "classification":
        prob = predict_proba(model, latest_X)[0]
        confidence = prob * 100
        direction = "UP" if prob >= 0.5 else "DOWN"
        prediction = prob
    else:
        direction = "UP" if y_pred > 0 else "DOWN"
        confidence = min(abs(y_pred) * 100, 100)
        prediction = y_pred

    return {
        "prediction": float(prediction),
        "direction": direction,
        "confidence": round(float(confidence), 2),
    }
