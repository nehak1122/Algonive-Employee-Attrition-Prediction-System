"""
Model Training Module for EAPS
Trains multiple classifiers, evaluates, and saves the best one.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import joblib

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data_preprocessing import prepare_data


def get_models():
    """Return dict of models to train."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, scale_pos_weight=3,
            use_label_encoder=False, eval_metric="logloss"
        ),
    }


def evaluate_model(model, X_test, y_test):
    """Evaluate a model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    return metrics, y_pred, y_proba


def get_feature_importance(model, feature_cols, model_name):
    """Extract feature importance from trained model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return {}

    importance_dict = dict(zip(feature_cols, importances.tolist()))
    # Sort descending
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    return importance_dict


def train_and_select_best(data_dir: str, artifacts_dir: str):
    """Train all models, compare, and save the best one."""
    os.makedirs(artifacts_dir, exist_ok=True)

    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(data_dir, artifacts_dir)

    models = get_models()
    results = {}
    best_model_name = None
    best_f1 = -1

    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
        results[name] = metrics

        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

        # Track best model by F1 score (better for imbalanced data)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = name

    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_model_name} (F1 Score: {best_f1:.4f})")
    print(f"{'=' * 60}")

    # Retrain best model and save
    best_model = models[best_model_name]

    # Save model
    joblib.dump(best_model, os.path.join(artifacts_dir, "best_model.pkl"))

    # Save feature importance
    importance = get_feature_importance(best_model, feature_cols, best_model_name)
    with open(os.path.join(artifacts_dir, "feature_importance.json"), "w") as f:
        json.dump(importance, f, indent=2)

    # Save model metadata
    metadata = {
        "best_model": best_model_name,
        "best_metrics": results[best_model_name],
        "all_results": results,
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(os.path.join(artifacts_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Print classification report for best model
    y_pred_best = best_model.predict(X_test)
    print(f"\nClassification Report ({best_model_name}):")
    print(classification_report(y_test, y_pred_best, target_names=["Stayed", "Left"]))

    cm = confusion_matrix(y_test, y_pred_best)
    print("Confusion Matrix:")
    print(cm)

    print(f"\n[INFO] All artifacts saved to: {artifacts_dir}")
    return best_model, metadata


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml", "artifacts")
    train_and_select_best(DATA_DIR, ARTIFACTS_DIR)
