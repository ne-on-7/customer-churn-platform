"""
Training pipeline for Customer Churn Intelligence Platform.
Trains all 5 models with 5-fold stratified cross-validation, saves best models.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.calibration import CalibratedClassifierCV

from src.data_processing import run_pipeline, get_scaler
from src.feature_engineering import add_engineered_features
from src.models import get_sklearn_models, PyTorchChurnClassifier


PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")


def evaluate_fold(model, X_val, y_val) -> dict:
    """Compute all metrics for a single fold."""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_proba),
    }


def train_all_models():
    """Train all models with cross-validation and save the best of each."""
    # Load and process data
    print("=" * 70)
    print("CUSTOMER CHURN INTELLIGENCE PLATFORM — TRAINING PIPELINE")
    print("=" * 70)

    X_train, X_test, y_train, y_test, scaler = run_pipeline(save=True)

    # Add engineered features
    print("\n[+] Adding engineered features...")
    X_train = add_engineered_features(X_train)
    X_test = add_engineered_features(X_test)

    # Update saved data with engineered features
    processed_dir = os.path.join(PROJECT_DIR, "data", "processed")
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)

    # Refit scaler with engineered features
    scaler = get_scaler(X_train)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(list(X_train.columns), os.path.join(MODELS_DIR, "feature_names.pkl"))

    # Scaled versions for models that need it
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Define models
    models = get_sklearn_models()
    models["Neural Network"] = PyTorchChurnClassifier(input_dim=X_train.shape[1], epochs=100)

    # Models that need scaled input
    needs_scaling = {"Logistic Regression", "Neural Network"}

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = {}

    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("=" * 70)

    for name, model in models.items():
        fold_metrics = []
        X_data = X_train_scaled if name in needs_scaling else X_train

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_train)):
            X_fold_train = X_data.iloc[train_idx]
            X_fold_val = X_data.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Clone model for each fold (sklearn models)
            if name == "Neural Network":
                fold_model = PyTorchChurnClassifier(input_dim=X_train.shape[1], epochs=100)
            else:
                from sklearn.base import clone
                fold_model = clone(model)

            fold_model.fit(X_fold_train, y_fold_train)
            metrics = evaluate_fold(fold_model, X_fold_val, y_fold_val)
            fold_metrics.append(metrics)

        # Average across folds
        avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        std_metrics = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        all_results[name] = {"mean": avg_metrics, "std": std_metrics}

        print(f"{name:<25} {avg_metrics['accuracy']:>10.4f} {avg_metrics['precision']:>10.4f} "
              f"{avg_metrics['recall']:>10.4f} {avg_metrics['f1']:>10.4f} {avg_metrics['roc_auc']:>10.4f}")

    print("=" * 70)

    # Train final models on full training set and save
    print("\n[+] Training final models on full training set...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, model in models.items():
        X_data = X_train_scaled if name in needs_scaling else X_train
        model.fit(X_data, y_train)

        if name == "Neural Network":
            import torch
            torch.save(model.model.state_dict(), os.path.join(MODELS_DIR, "neural_network.pt"))
            joblib.dump({"input_dim": X_train.shape[1], "epochs": 100}, os.path.join(MODELS_DIR, "nn_params.pkl"))
        else:
            safe_name = name.lower().replace(" ", "_")
            joblib.dump(model, os.path.join(MODELS_DIR, f"{safe_name}.pkl"))

        print(f"    Saved: {name}")

    # Save CV results
    with open(os.path.join(MODELS_DIR, "cv_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print test set performance
    print("\n" + "=" * 70)
    print("TEST SET PERFORMANCE (held-out 20%)")
    print("=" * 70)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 70)

    test_results = {}
    for name, model in models.items():
        X_data = X_test_scaled if name in needs_scaling else X_test
        metrics = evaluate_fold(model, X_data, y_test)
        test_results[name] = metrics
        print(f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['roc_auc']:>10.4f}")

    print("=" * 70)

    with open(os.path.join(MODELS_DIR, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    # Identify best model
    best_name = max(test_results, key=lambda k: test_results[k]["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best_name} ({test_results[best_name]['roc_auc']:.4f})")
    with open(os.path.join(MODELS_DIR, "best_model.txt"), "w") as f:
        f.write(best_name)

    # Calibrate the best model so predicted probabilities reflect true risk
    print("\n[+] Calibrating best model probabilities (Platt scaling)...")
    best_model = models[best_name]
    X_cal = X_train_scaled if best_name in needs_scaling else X_train
    calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv=5)
    calibrated.fit(X_cal, y_train)

    safe_best = best_name.lower().replace(" ", "_")
    joblib.dump(calibrated, os.path.join(MODELS_DIR, f"{safe_best}_calibrated.pkl"))
    print(f"    Saved calibrated model: {safe_best}_calibrated.pkl")

    # Show calibration effect on test set
    X_test_cal = X_test_scaled if best_name in needs_scaling else X_test
    raw_probas = best_model.predict_proba(X_test_cal)[:, 1]
    cal_probas = calibrated.predict_proba(X_test_cal)[:, 1]
    print(f"    Raw  probabilities — mean: {raw_probas.mean():.3f}, median: {np.median(raw_probas):.3f}")
    print(f"    Calibrated probas  — mean: {cal_probas.mean():.3f}, median: {np.median(cal_probas):.3f}")

    return models, all_results, test_results


if __name__ == "__main__":
    train_all_models()
