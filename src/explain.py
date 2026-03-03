"""
Explainability module for Customer Churn Intelligence Platform.
Uses SHAP for global and local model interpretability.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from src.data_processing import run_pipeline
from src.feature_engineering import add_engineered_features
from src.evaluate import load_trained_models

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")


def get_shap_explainer(model, model_name: str, X_background: pd.DataFrame):
    """Create the appropriate SHAP explainer for a given model."""
    if model_name in ("XGBoost", "Random Forest", "Gradient Boosting"):
        return shap.TreeExplainer(model)
    else:
        return shap.KernelExplainer(model.predict_proba, shap.sample(X_background, 100))


def global_explanation(model, model_name: str, X_train: pd.DataFrame,
                       X_test: pd.DataFrame, save_path: str = None):
    """Generate global SHAP summary plot."""
    explainer = get_shap_explainer(model, model_name, X_train)

    if model_name in ("XGBoost", "Random Forest", "Gradient Boosting"):
        shap_values = explainer.shap_values(X_test)
    else:
        shap_values = explainer.shap_values(X_test[:100])
        X_test = X_test[:100]

    # Handle (n_samples, n_features, n_classes) → take churn class
    sv = np.array(shap_values)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    elif isinstance(shap_values, list) and len(shap_values) > 1:
        sv = np.array(shap_values[1])

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(sv, X_test, show=False, max_display=15)
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return shap_values


def local_explanation(model, model_name: str, X_train: pd.DataFrame,
                      X_instance: pd.DataFrame, customer_idx: int = 0):
    """Generate SHAP explanation for a single customer."""
    explainer = get_shap_explainer(model, model_name, X_train)

    if isinstance(X_instance, pd.Series):
        X_instance = X_instance.to_frame().T

    shap_values = explainer.shap_values(X_instance)

    sv = np.array(shap_values)

    # Handle shape (n_samples, n_features, n_classes) from KernelExplainer
    if sv.ndim == 3:
        # Take first sample, churn class (index 1)
        sv = sv[0, :, 1]
    elif isinstance(shap_values, list) and len(shap_values) > 1:
        # Tree models return [class_0_shap, class_1_shap]
        sv = np.array(shap_values[1])
        if sv.ndim > 1:
            sv = sv[0]
    elif sv.ndim == 2:
        sv = sv[0]

    return sv, X_instance.columns.tolist()


def get_top_reasons(shap_values_single: np.ndarray, feature_names: list, top_n: int = 3) -> list:
    """Get top N human-readable reasons for a churn prediction."""
    # Ensure shap values are flat scalars
    sv_flat = np.array(shap_values_single).flatten()
    feature_impacts = list(zip(feature_names, [float(v) for v in sv_flat]))

    # Sort by absolute SHAP value (most impactful first)
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feat, val in feature_impacts[:top_n]:
        direction = "increases" if val > 0 else "decreases"
        clean_name = feat.replace("_", " ").replace("Encoded", "").strip()
        reasons.append(f"{clean_name} ({direction} churn risk)")

    return reasons


def run_explanation():
    """Run full explainability pipeline."""
    print("=" * 70)
    print("CUSTOMER CHURN INTELLIGENCE PLATFORM — EXPLAINABILITY")
    print("=" * 70)

    # Load data
    X_train, X_test, y_train, y_test, _ = run_pipeline(save=False)
    X_train = add_engineered_features(X_train)
    X_test = add_engineered_features(X_test)

    # Load best model
    best_name = open(os.path.join(MODELS_DIR, "best_model.txt")).read().strip()
    models = load_trained_models(input_dim=X_test.shape[1])
    model = models[best_name]
    print(f"\nExplaining: {best_name}")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Global explanation
    print("\n[1/2] Generating global SHAP summary...")
    global_explanation(model, best_name, X_train, X_test,
                       save_path=os.path.join(PLOTS_DIR, "shap_global.png"))

    # Local explanations for sample customers
    print("\n[2/2] Generating local explanations for sample customers...")

    # Find a few interesting cases
    needs_scaling = {"Logistic Regression", "Neural Network"}
    if best_name in needs_scaling:
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        X_eval = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    else:
        X_eval = X_test

    y_proba = model.predict_proba(X_eval)[:, 1]

    # High risk customer
    high_risk_idx = np.argmax(y_proba)
    sv, feat_names = local_explanation(model, best_name, X_train, X_test.iloc[[high_risk_idx]])
    reasons = get_top_reasons(sv, feat_names)

    print(f"\n  HIGH RISK Customer (probability: {y_proba[high_risk_idx]:.2%}):")
    print(f"  Actual: {'Churned' if y_test.iloc[high_risk_idx] == 1 else 'Retained'}")
    for i, reason in enumerate(reasons, 1):
        print(f"    {i}. {reason}")

    # Low risk customer
    low_risk_idx = np.argmin(y_proba)
    sv, feat_names = local_explanation(model, best_name, X_train, X_test.iloc[[low_risk_idx]])
    reasons = get_top_reasons(sv, feat_names)

    print(f"\n  LOW RISK Customer (probability: {y_proba[low_risk_idx]:.2%}):")
    print(f"  Actual: {'Churned' if y_test.iloc[low_risk_idx] == 1 else 'Retained'}")
    for i, reason in enumerate(reasons, 1):
        print(f"    {i}. {reason}")

    print(f"\nPlots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    run_explanation()
