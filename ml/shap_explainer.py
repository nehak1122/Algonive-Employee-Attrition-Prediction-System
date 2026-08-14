"""
SHAP Explainability Module for EAPS
Explains individual and global predictions using SHAP, so HR can see
*why* the model flagged a specific employee, not just the flag itself.
"""

import os
import numpy as np
import pandas as pd
import shap


def build_explainer(model, background_data: pd.DataFrame):
    """Build a SHAP explainer appropriate for the given model type.

    Tree-based models (RandomForest, XGBoost) use the fast, exact TreeExplainer.
    Everything else (e.g. LogisticRegression) falls back to a KernelExplainer on
    a small background sample, since it works for any model but is much slower.
    """
    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        return shap.TreeExplainer(model), "tree"

    background_sample = shap.sample(background_data, min(50, len(background_data)), random_state=42)
    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    return shap.KernelExplainer(predict_fn, background_sample), "kernel"


def _to_positive_class_values(raw_shap_values):
    """Normalize SHAP output across explainer/model types to a 1D array for the positive class."""
    if isinstance(raw_shap_values, list):
        # Older SHAP API: list of arrays, one per class
        return np.array(raw_shap_values[1]) if len(raw_shap_values) > 1 else np.array(raw_shap_values[0])
    arr = np.array(raw_shap_values)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        return arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def explain_instance(explainer, kind: str, X_row: pd.DataFrame, feature_cols):
    """Return the top contributing features (with signed SHAP value) for a single row.

    A positive value pushes the prediction toward "will leave"; negative pushes
    toward "will stay".
    """
    raw = explainer.shap_values(X_row)
    values = _to_positive_class_values(raw)[0]

    contributions = sorted(
        zip(feature_cols, values.tolist()),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    return [
        {"feature": name, "impact": round(float(val), 4),
         "direction": "increases_risk" if val > 0 else "decreases_risk"}
        for name, val in contributions
    ]


def global_importance(explainer, kind: str, X_sample: pd.DataFrame, feature_cols):
    """Mean |SHAP value| per feature across a sample of rows — the global picture."""
    raw = explainer.shap_values(X_sample)
    values = _to_positive_class_values(raw)
    mean_abs = np.abs(values).mean(axis=0)
    importance = dict(zip(feature_cols, mean_abs.tolist()))
    return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
