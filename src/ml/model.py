# src/ml/model.py

from typing import Literal
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


ModelType = Literal[
    "linear_regression",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
]


def build_model(
    model_type: ModelType,
    task: Literal["regression", "classification"],
):
    """
    Factory for ML models.
    """

    if model_type == "linear_regression":
        return LinearRegression()

    if model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
        )

    if model_type == "random_forest":
        if task == "regression":
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
            )
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )

    if model_type == "gradient_boosting":
        if task == "regression":
            return GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
        return GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def train_model(
    model,
    X_train,
    y_train,
):
    """
    Fit ML model.
    """
    model.fit(X_train, y_train)
    return model


def predict(
    model,
    X,
):
    """
    Predict using trained model.
    """
    return model.predict(X)


def predict_proba(
    model,
    X,
):
    """
    Predict probability (classification only).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise ValueError("Model does not support probability prediction")


def save_model(
    model,
    name: str,
):
    """
    Persist model to disk.
    """
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(
    name: str,
):
    """
    Load persisted model.
    """
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)