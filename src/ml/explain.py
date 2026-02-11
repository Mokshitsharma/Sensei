# src/ml/explain.py

from typing import Dict, List
import numpy as np

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def explain_global(
    model,
    X,
    feature_names: List[str],
    max_features: int = 10,
) -> Dict[str, float]:
    """
    Global feature importance explanation.

    Returns:
        {feature_name: importance}
    """

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()

    elif _SHAP_AVAILABLE:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        importances = np.abs(shap_values.values).mean(axis=0)

    else:
        raise ValueError("Model type not supported for explanation")

    importance_map = dict(
        sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:max_features]
    )

    return {k: float(v) for k, v in importance_map.items()}


def explain_prediction(
    model,
    x_row: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Explain a single prediction.

    Returns:
        {feature_name: contribution}
    """

    if _SHAP_AVAILABLE:
        explainer = shap.Explainer(model, x_row.reshape(1, -1))
        shap_values = explainer(x_row.reshape(1, -1))
        contribs = shap_values.values[0]

    elif hasattr(model, "coef_"):
        contribs = model.coef_.ravel() * x_row

    else:
        raise ValueError("Local explanation not supported for this model")

    return {
        name: float(val)
        for name, val in zip(feature_names, contribs)
    }
