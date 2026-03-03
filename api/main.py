"""
FastAPI REST API for Customer Churn Intelligence Platform.
Serves churn predictions with SHAP explanations.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_DIR)

from src.feature_engineering import add_engineered_features
from src.evaluate import load_trained_models
from src.explain import local_explanation, get_top_reasons

app = FastAPI(
    title="Customer Churn Intelligence API",
    description="Predict customer churn probability with explainable AI",
    version="1.0.0",
)

# Global state for loaded models
_state = {}
MODELS_DIR = os.path.join(PROJECT_DIR, "models")


class CustomerInput(BaseModel):
    """Input schema for a single customer prediction."""
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: int = Field(..., ge=0, le=1)
    Dependents: int = Field(..., ge=0, le=1)
    tenure: int = Field(..., ge=0, le=72)
    PhoneService: int = Field(..., ge=0, le=1)
    MultipleLines: int = Field(..., ge=0, le=1)
    OnlineSecurity: int = Field(..., ge=0, le=1)
    OnlineBackup: int = Field(..., ge=0, le=1)
    DeviceProtection: int = Field(..., ge=0, le=1)
    TechSupport: int = Field(..., ge=0, le=1)
    StreamingTV: int = Field(..., ge=0, le=1)
    StreamingMovies: int = Field(..., ge=0, le=1)
    PaperlessBilling: int = Field(..., ge=0, le=1)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    InternetService_Fiber_optic: int = Field(0, ge=0, le=1, alias="InternetService_Fiber optic")
    InternetService_No: int = Field(0, ge=0, le=1)
    Contract_One_year: int = Field(0, ge=0, le=1, alias="Contract_One year")
    Contract_Two_year: int = Field(0, ge=0, le=1, alias="Contract_Two year")
    PaymentMethod_Credit_card: int = Field(0, ge=0, le=1, alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(0, ge=0, le=1, alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(0, ge=0, le=1, alias="PaymentMethod_Mailed check")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_tier: str
    top_reasons: list[str]
    model_used: str


def load_state():
    """Load models and artifacts on startup."""
    if _state:
        return

    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    best_name = open(os.path.join(MODELS_DIR, "best_model.txt")).read().strip()
    models = load_trained_models(input_dim=len(feature_names))

    _state["feature_names"] = feature_names
    _state["scaler"] = scaler
    _state["best_name"] = best_name
    _state["models"] = models


@app.on_event("startup")
async def startup():
    try:
        load_state()
    except Exception as e:
        print(f"Warning: Could not load models on startup: {e}")
        print("Run 'python -m src.train' first to train models.")


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": len(_state.get("models", {}))}


@app.get("/models")
def list_models():
    """Return performance summary of all trained models."""
    results_path = os.path.join(MODELS_DIR, "test_results.json")
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="No test results found. Train models first.")

    with open(results_path) as f:
        results = json.load(f)

    best_name = _state.get("best_name", "Unknown")
    return {"best_model": best_name, "results": results}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    """Predict churn probability for a single customer with explanations."""
    if not _state:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training pipeline first.")

    feature_names = _state["feature_names"]
    scaler = _state["scaler"]
    best_name = _state["best_name"]
    model = _state["models"][best_name]

    # Build feature DataFrame
    input_dict = customer.model_dump(by_alias=True)
    df = pd.DataFrame([input_dict])

    # Rename columns to match training data
    rename_map = {
        "InternetService_Fiber optic": "InternetService_Fiber optic",
        "Contract_One year": "Contract_One year",
        "Contract_Two year": "Contract_Two year",
        "PaymentMethod_Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check": "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check": "PaymentMethod_Mailed check",
    }
    df = df.rename(columns=rename_map)

    # Add engineered features
    df = add_engineered_features(df)

    # Ensure all expected features are present in the right order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale if needed
    needs_scaling = {"Logistic Regression", "Neural Network"}
    if best_name in needs_scaling:
        X_input = pd.DataFrame(scaler.transform(df), columns=df.columns)
    else:
        X_input = df

    # Predict
    proba = model.predict_proba(X_input)[0, 1]

    # Risk tier
    if proba >= 0.7:
        risk_tier = "High"
    elif proba >= 0.4:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    # SHAP explanation
    try:
        # Use a small background sample for SHAP
        processed_dir = os.path.join(PROJECT_DIR, "data", "processed")
        X_train_bg = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
        sv, feat_names = local_explanation(model, best_name, X_train_bg, df)
        top_reasons = get_top_reasons(sv, feat_names, top_n=3)
    except Exception:
        top_reasons = ["Explanation unavailable"]

    return PredictionResponse(
        churn_probability=round(float(proba), 4),
        risk_tier=risk_tier,
        top_reasons=top_reasons,
        model_used=best_name,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
