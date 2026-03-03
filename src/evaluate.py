"""
Evaluation module for Customer Churn Intelligence Platform.
Generates ROC curves, PR curves, confusion matrices, and business impact metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, classification_report)

from src.data_processing import run_pipeline, get_scaler
from src.feature_engineering import add_engineered_features
from src.models import get_sklearn_models, PyTorchChurnClassifier

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")


def load_trained_models(input_dim: int) -> dict:
    """Load all saved models."""
    models = {}
    sklearn_names = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
    }
    for name, filename in sklearn_names.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    # Load Neural Network
    import torch
    nn_path = os.path.join(MODELS_DIR, "neural_network.pt")
    if os.path.exists(nn_path):
        nn_model = PyTorchChurnClassifier(input_dim=input_dim, epochs=100)
        nn_model.model.load_state_dict(torch.load(nn_path, weights_only=True))
        nn_model.model.eval()
        models["Neural Network"] = nn_model

    return models


def plot_roc_curves(models: dict, X_test, y_test, save_path: str = None):
    """Plot ROC curves for all models overlaid."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    needs_scaling = {"Logistic Regression", "Neural Network"}

    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    for (name, model), color in zip(models.items(), colors):
        X_data = X_test_scaled if name in needs_scaling else X_test
        y_proba = model.predict_proba(X_data)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_precision_recall_curves(models: dict, X_test, y_test, save_path: str = None):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    needs_scaling = {"Logistic Regression", "Neural Network"}

    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    for (name, model), color in zip(models.items(), colors):
        X_data = X_test_scaled if name in needs_scaling else X_test
        y_proba = model.predict_proba(X_data)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=color, lw=2, label=f'{name} (AUC = {pr_auc:.3f})')

    baseline = y_test.mean()
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — All Models', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_confusion_matrices(models: dict, X_test, y_test, save_path: str = None):
    """Plot confusion matrices for all models in a grid."""
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    needs_scaling = {"Logistic Regression", "Neural Network"}

    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    for idx, (name, model) in enumerate(models.items()):
        X_data = X_test_scaled if name in needs_scaling else X_test
        y_pred = model.predict(X_data)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Retained', 'Churned'],
                    yticklabels=['Retained', 'Churned'])
        axes[idx].set_title(name, fontsize=11)
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')

    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Confusion Matrices — All Models', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def compute_business_impact(model, X_test, y_test, avg_monthly_revenue: float = None,
                            retention_cost: float = 20.0, months_saved: int = 6):
    """
    Estimate revenue impact of using the model for proactive retention.

    Assumes:
    - Each caught true positive saves avg_monthly_revenue × months_saved
    - Each retention offer costs retention_cost
    """
    if avg_monthly_revenue is None:
        avg_monthly_revenue = 65.0  # approximate from dataset mean

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    revenue_saved = tp * avg_monthly_revenue * months_saved
    retention_spend = (tp + fp) * retention_cost  # offers sent to all predicted churners
    missed_revenue = fn * avg_monthly_revenue * months_saved
    net_benefit = revenue_saved - retention_spend

    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "revenue_saved": round(revenue_saved, 2),
        "retention_spend": round(retention_spend, 2),
        "missed_revenue": round(missed_revenue, 2),
        "net_benefit": round(net_benefit, 2),
        "roi_percent": round((net_benefit / retention_spend) * 100, 1) if retention_spend > 0 else 0,
    }


def run_evaluation():
    """Run full evaluation pipeline."""
    print("=" * 70)
    print("CUSTOMER CHURN INTELLIGENCE PLATFORM — EVALUATION")
    print("=" * 70)

    # Load data
    X_train, X_test, y_train, y_test, _ = run_pipeline(save=False)
    X_train = add_engineered_features(X_train)
    X_test = add_engineered_features(X_test)

    # Load models
    models = load_trained_models(input_dim=X_test.shape[1])
    print(f"\nLoaded {len(models)} models: {list(models.keys())}")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Generate plots
    print("\n[1/4] Generating ROC curves...")
    plot_roc_curves(models, X_test, y_test, os.path.join(PLOTS_DIR, "roc_curves.png"))

    print("[2/4] Generating Precision-Recall curves...")
    plot_precision_recall_curves(models, X_test, y_test, os.path.join(PLOTS_DIR, "pr_curves.png"))

    print("[3/4] Generating confusion matrices...")
    plot_confusion_matrices(models, X_test, y_test, os.path.join(PLOTS_DIR, "confusion_matrices.png"))

    # Business impact
    print("[4/4] Computing business impact...")
    best_name = open(os.path.join(MODELS_DIR, "best_model.txt")).read().strip()
    best_model = models[best_name]

    needs_scaling = {"Logistic Regression", "Neural Network"}
    if best_name in needs_scaling:
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        X_eval = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    else:
        X_eval = X_test

    impact = compute_business_impact(best_model, X_eval, y_test)

    print(f"\n{'=' * 50}")
    print(f"BUSINESS IMPACT ANALYSIS ({best_name})")
    print(f"{'=' * 50}")
    print(f"  Customers correctly identified as churning: {impact['true_positives']}")
    print(f"  Customers missed (false negatives):        {impact['false_negatives']}")
    print(f"  False alarms (false positives):             {impact['false_positives']}")
    print(f"  Revenue saved (est.):                       ${impact['revenue_saved']:,.2f}")
    print(f"  Retention campaign cost:                    ${impact['retention_spend']:,.2f}")
    print(f"  Revenue still lost (missed churners):       ${impact['missed_revenue']:,.2f}")
    print(f"  Net benefit:                                ${impact['net_benefit']:,.2f}")
    print(f"  ROI:                                        {impact['roi_percent']}%")
    print(f"{'=' * 50}")

    with open(os.path.join(MODELS_DIR, "business_impact.json"), "w") as f:
        json.dump(impact, f, indent=2)

    print(f"\nPlots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    run_evaluation()
