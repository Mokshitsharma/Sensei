# src/ml/shap_explain.py
"""
SHAP-based explainability for the ML ensemble.

Uses TreeExplainer for RandomForest / GradientBoosting (exact, fast).
Falls back to KernelExplainer for unsupported model types.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
) -> Optional[Dict[str, float]]:
    """
    Compute SHAP values for a single prediction row (X_explain).

    Args:
        model:          Trained sklearn model (RF / GBM / LogReg supported)
        X_background:   Background dataset for KernelExplainer fallback
                        (use a small representative sample, e.g. 50–100 rows)
        X_explain:      Single row to explain, shape (1, n_features)
        feature_names:  Column names matching X_explain

    Returns:
        Dict mapping feature name → SHAP value (float), or None if SHAP
        is unavailable.
    """
    if not _SHAP_AVAILABLE:
        return None

    explainer = _build_explainer(model, X_background)
    raw = explainer(X_explain)

    # shap_values.values shape: (1, n_features) for regression
    # or (1, n_features, n_classes) for multi-class — take class-1 slice
    vals = raw.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]          # class = UP (positive class)
    shap_row = vals[0]                # shape: (n_features,)

    return {
        name: float(v)
        for name, v in zip(feature_names, shap_row)
    }


def rank_features_by_impact(
    shap_values: Dict[str, float],
    top_n: int = 9,
) -> List[Dict]:
    """
    Sort features by absolute SHAP impact and return ranked list.

    Returns list of dicts:
        [{"feature": str, "shap": float, "direction": "bullish"|"bearish"}, ...]
    """
    ranked = sorted(
        shap_values.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:top_n]

    return [
        {
            "feature": feat,
            "shap": val,
            "direction": "bullish" if val > 0 else "bearish",
        }
        for feat, val in ranked
    ]


def global_feature_importance(
    model,
    X: np.ndarray,
    feature_names: List[str],
    top_n: int = 9,
) -> Dict[str, float]:
    """
    Mean absolute SHAP across the full dataset — global importance.

    Returns {feature: mean_abs_shap} sorted descending.
    """
    if not _SHAP_AVAILABLE:
        # Fallback: tree-native importances
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            return dict(
                sorted(
                    zip(feature_names, imp),
                    key=lambda x: x[1],
                    reverse=True,
                )[:top_n]
            )
        return {}

    explainer = _build_explainer(model, X)
    raw = explainer(X)
    vals = raw.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    mean_abs = np.abs(vals).mean(axis=0)
    ranked = sorted(
        zip(feature_names, mean_abs),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    return {k: float(v) for k, v in ranked}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_explainer(model, X_background: np.ndarray):
    """
    Pick the best SHAP explainer for the given model type.

    Priority:
        TreeExplainer  → RF, GBM (exact, fast)
        LinearExplainer → LogReg / LinearSVC
        KernelExplainer → anything else (slow, approximate)
    """
    model_cls = type(model).__name__

    tree_types = {
        "RandomForestClassifier",
        "RandomForestRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
    }

    linear_types = {
        "LogisticRegression",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "LinearSVC",
        "SGDClassifier",
    }

    if model_cls in tree_types:
        return shap.TreeExplainer(model)

    if model_cls in linear_types:
        # LinearExplainer needs a masker / background data
        masker = shap.maskers.Independent(X_background, max_samples=100)
        return shap.LinearExplainer(model, masker)

    # Generic fallback
    background = shap.sample(X_background, 50)
    return shap.KernelExplainer(model.predict, background)
