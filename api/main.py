"""
FastAPI REST API for Customer Churn Intelligence Platform.
Serves churn predictions with SHAP explanations, data exploration,
business impact analysis, and AI chat with Claude.
"""

import os
import sys
import json
import uuid
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File as FastAPIFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import io

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_DIR)

from src.feature_engineering import add_engineered_features
from src.evaluate import load_trained_models, compute_business_impact
from src.explain import local_explanation, get_top_reasons
from src.data_processing import load_raw_data
from src.experimentation import (
    create_experiment as run_experiment,
    list_experiments as get_experiments,
    load_experiment,
    compute_power_analysis,
)

app = FastAPI(
    title="Customer Churn Intelligence API",
    description="Predict customer churn probability with explainable AI",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for loaded models
_state = {}
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
HISTORY_FILE = os.path.join(DATA_DIR, "prediction_history.json")

# Centralized risk tier thresholds (used across prediction, high-risk, and batch endpoints)
RISK_THRESHOLDS = {"High": 0.5, "Medium": 0.2}


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


class BusinessImpactRequest(BaseModel):
    avg_monthly_revenue: float = Field(65.0, ge=1.0)
    retention_cost: float = Field(20.0, ge=0.0)
    months_saved: int = Field(6, ge=1, le=24)


class ChatRequest(BaseModel):
    messages: List[dict]


class ExperimentCreate(BaseModel):
    name: str = Field(..., description="Human-readable experiment name")
    intervention_type: str = Field(..., description="discount | personalized_email | service_upgrade | loyalty_program")
    intervention_description: str = Field("", description="Free-text description")
    expected_effect_size: float = Field(0.15, ge=0.01, le=0.50)
    cost_per_customer: float = Field(20.0, ge=0.0)
    risk_tiers: List[str] = Field(["High"], description="Target risk tiers")
    feature_filters: Optional[dict] = None
    split_ratio: float = Field(0.5, ge=0.1, le=0.9)
    significance_level: float = Field(0.05, ge=0.01, le=0.10)
    power: float = Field(0.80, ge=0.70, le=0.99)
    random_seed: int = Field(42)
    avg_monthly_revenue: float = Field(65.0, ge=1.0)
    months_saved: int = Field(6, ge=1, le=24)


class PowerAnalysisRequest(BaseModel):
    baseline_churn_rate: float = Field(..., ge=0.01, le=0.99)
    minimum_detectable_effect: float = Field(..., ge=0.01, le=0.50)
    alpha: float = Field(0.05, ge=0.01, le=0.10)
    power: float = Field(0.80, ge=0.70, le=0.99)
    eligible_population: int = Field(0, ge=0)


def _clamp_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp numeric inputs to training data range to prevent nonsensical predictions."""
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = df["MonthlyCharges"].clip(18, 120)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].clip(0, 8700)
    if "tenure" in df.columns:
        df["tenure"] = df["tenure"].clip(0, 72)
    return df


def _to_python(val):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def load_state():
    """Load models and artifacts on startup."""
    if _state:
        return

    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    best_name = open(os.path.join(MODELS_DIR, "best_model.txt")).read().strip()
    models = load_trained_models(input_dim=len(feature_names))

    # Load calibrated model if available (produces well-calibrated probabilities)
    safe_best = best_name.lower().replace(" ", "_")
    calibrated_path = os.path.join(MODELS_DIR, f"{safe_best}_calibrated.pkl")
    calibrated_model = None
    if os.path.exists(calibrated_path):
        calibrated_model = joblib.load(calibrated_path)
        print(f"Loaded calibrated model: {safe_best}_calibrated.pkl")

    # Cache a small background sample for SHAP
    processed_dir = os.path.join(DATA_DIR, "processed")
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    bg_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)

    # Cache test data for business impact
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()

    # Cache raw data
    df_raw = load_raw_data()

    _state["feature_names"] = feature_names
    _state["scaler"] = scaler
    _state["best_name"] = best_name
    _state["models"] = models
    _state["calibrated_model"] = calibrated_model
    _state["bg_sample"] = bg_sample
    _state["X_test"] = X_test
    _state["y_test"] = y_test
    _state["df_raw"] = df_raw


@app.on_event("startup")
async def startup():
    try:
        load_state()
        print(f"Models loaded successfully: {list(_state.get('models', {}).keys())}")
    except Exception as e:
        import traceback
        print(f"WARNING: Could not load models on startup: {e}")
        traceback.print_exc()
        print("Run 'python -m src.train' first to train models.")


def _predict_single(input_dict: dict) -> dict:
    """Internal prediction logic shared by predict and batch endpoints."""
    feature_names = _state["feature_names"]
    scaler = _state["scaler"]
    best_name = _state["best_name"]
    model = _state["models"][best_name]
    calibrated_model = _state.get("calibrated_model")

    df = pd.DataFrame([input_dict])
    df = _clamp_inputs(df)
    df = add_engineered_features(df)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    needs_scaling = {"Logistic Regression", "Neural Network"}
    if best_name in needs_scaling:
        X_input = pd.DataFrame(scaler.transform(df), columns=df.columns)
    else:
        X_input = df

    # Use calibrated model for well-calibrated probabilities, fall back to raw model
    predict_model = calibrated_model if calibrated_model is not None else model
    proba = predict_model.predict_proba(X_input)[0, 1]

    if proba >= RISK_THRESHOLDS["High"]:
        risk_tier = "High"
    elif proba >= RISK_THRESHOLDS["Medium"]:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    try:
        sv, feat_names = local_explanation(model, best_name, _state["bg_sample"], df)
        top_reasons = get_top_reasons(sv, feat_names, top_n=3)
    except Exception as e:
        print(f"Warning: SHAP explanation failed: {e}")
        top_reasons = ["Explanation unavailable"]

    return {
        "churn_probability": round(float(proba), 4),
        "risk_tier": risk_tier,
        "top_reasons": top_reasons,
        "model_used": best_name,
    }


def _save_prediction(input_dict: dict, result: dict):
    """Save prediction to history file."""
    try:
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "churn_probability": result["churn_probability"],
            "risk_tier": result["risk_tier"],
            "top_reasons": result["top_reasons"],
            "model_used": result["model_used"],
            "inputs": {k: _to_python(v) for k, v in input_dict.items()},
        }
        history = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE) as f:
                history = json.load(f)
        history.insert(0, entry)
        history = history[:500]  # Keep last 500
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save prediction history: {e}")


# ──────────────────────────────────────────────────────────────
# CORE ENDPOINTS
# ──────────────────────────────────────────────────────────────

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

    input_dict = customer.model_dump(by_alias=True)
    result = _predict_single(input_dict)
    _save_prediction(input_dict, result)
    return PredictionResponse(**result)


# ──────────────────────────────────────────────────────────────
# DATA EXPLORER ENDPOINTS
# ──────────────────────────────────────────────────────────────

@app.get("/data/overview")
def data_overview():
    """Dataset overview statistics."""
    if "df_raw" not in _state:
        raise HTTPException(status_code=503, detail="Data not loaded.")
    df = _state["df_raw"]
    churn_rate = (df["Churn"] == "Yes").mean()
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "churn_rate": round(float(churn_rate), 4),
        "features": list(df.columns),
    }


@app.get("/data/churn-rates")
def churn_rates(group_by: str = "Contract"):
    """Churn rate grouped by a categorical feature."""
    if "df_raw" not in _state:
        raise HTTPException(status_code=503, detail="Data not loaded.")
    df = _state["df_raw"]
    if group_by not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{group_by}' not found.")

    rates = df.groupby(group_by)["Churn"].apply(lambda x: (x == "Yes").mean()).reset_index()
    rates.columns = [group_by, "churn_rate"]
    counts = df.groupby(group_by).size().reset_index(name="count")
    merged = rates.merge(counts, on=group_by)
    return [
        {"group": str(row[group_by]), "churn_rate": round(float(row["churn_rate"]), 4), "count": int(row["count"])}
        for _, row in merged.iterrows()
    ]


@app.get("/data/distribution")
def feature_distribution(feature: str = "MonthlyCharges"):
    """Histogram data for a numerical feature split by churn."""
    if "df_raw" not in _state:
        raise HTTPException(status_code=503, detail="Data not loaded.")
    df = _state["df_raw"]
    if feature not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{feature}' not found.")

    col = pd.to_numeric(df[feature], errors="coerce").dropna()
    churned = col[df.loc[col.index, "Churn"] == "Yes"]
    retained = col[df.loc[col.index, "Churn"] == "No"]

    bins = np.linspace(col.min(), col.max(), 41)
    hist_churned, _ = np.histogram(churned, bins=bins)
    hist_retained, _ = np.histogram(retained, bins=bins)
    bin_centers = ((bins[:-1] + bins[1:]) / 2).tolist()

    return {
        "bins": [round(b, 2) for b in bin_centers],
        "churned": hist_churned.tolist(),
        "retained": hist_retained.tolist(),
    }


# ──────────────────────────────────────────────────────────────
# BUSINESS IMPACT
# ──────────────────────────────────────────────────────────────

@app.post("/business-impact")
def business_impact(req: BusinessImpactRequest):
    """Compute business impact of using the model for retention."""
    if not _state:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    best_name = _state["best_name"]
    model = _state["models"][best_name]
    X_test = _state["X_test"]
    y_test = _state["y_test"]

    needs_scaling = {"Logistic Regression", "Neural Network"}
    if best_name in needs_scaling:
        X_eval = pd.DataFrame(
            _state["scaler"].transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
    else:
        X_eval = X_test

    impact = compute_business_impact(
        model, X_eval, y_test,
        req.avg_monthly_revenue, req.retention_cost, req.months_saved,
    )
    return impact


# ──────────────────────────────────────────────────────────────
# HIGH RISK CUSTOMERS
# ──────────────────────────────────────────────────────────────

@app.get("/customers/high-risk")
def high_risk_customers(limit: int = 20):
    """Get top-N customers most likely to churn."""
    if not _state:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    df_raw = _state["df_raw"]
    best_name = _state["best_name"]
    model = _state["models"][best_name]
    feature_names = _state["feature_names"]
    scaler = _state["scaler"]

    # Encode the raw data for prediction
    from src.data_processing import encode_features
    df_encoded = encode_features(df_raw.copy())
    df_encoded = _clamp_inputs(df_encoded)
    df_encoded = add_engineered_features(df_encoded)

    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    X = df_encoded[feature_names]

    needs_scaling = {"Logistic Regression", "Neural Network"}
    if best_name in needs_scaling:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    else:
        X_scaled = X

    calibrated_model = _state.get("calibrated_model")
    predict_model = calibrated_model if calibrated_model is not None else model
    probas = predict_model.predict_proba(X_scaled)[:, 1]
    df_raw_copy = df_raw.copy()
    df_raw_copy["churn_probability"] = probas
    df_raw_copy["risk_tier"] = pd.cut(
        probas, bins=[0, RISK_THRESHOLDS["Medium"], RISK_THRESHOLDS["High"], 1.01],
        labels=["Low", "Medium", "High"],
    )

    top = df_raw_copy.nlargest(limit, "churn_probability")

    # Pre-compute feature means for retained customers (fast heuristic instead of SHAP)
    retained_mask = df_raw["Churn"] == "No"
    feature_means = X.loc[retained_mask.values].mean()

    results = []
    for idx, row in top.iterrows():
        try:
            deviations = (X.loc[idx] - feature_means).abs()
            top_feat = deviations.idxmax()
            direction = "increases" if float(X.loc[idx, top_feat]) > float(feature_means[top_feat]) else "decreases"
            top_reason = f"{top_feat.replace('_', ' ')} ({direction} churn risk)"
        except Exception:
            top_reason = "N/A"

        results.append({
            "customer_id": int(idx),
            "churn_probability": round(float(row["churn_probability"]), 4),
            "risk_tier": str(row["risk_tier"]),
            "top_reason": top_reason,
            "monthly_charges": float(row.get("MonthlyCharges", 0)),
            "tenure": int(row.get("tenure", 0)),
            "contract": str(row.get("Contract", "Unknown")),
        })

    return results


# ──────────────────────────────────────────────────────────────
# BATCH PREDICTION
# ──────────────────────────────────────────────────────────────

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = FastAPIFile(...)):
    """Upload a CSV and get predictions for all rows."""
    if not _state:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    feature_names = _state["feature_names"]
    best_name = _state["best_name"]
    model = _state["models"][best_name]
    calibrated_model = _state.get("calibrated_model")
    scaler = _state["scaler"]

    df_clamped = _clamp_inputs(df.copy())
    df_feat = add_engineered_features(df_clamped)
    for col in feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0
    X = df_feat[feature_names]

    needs_scaling = {"Logistic Regression", "Neural Network"}
    if best_name in needs_scaling:
        X_input = pd.DataFrame(scaler.transform(X), columns=X.columns)
    else:
        X_input = X

    predict_model = calibrated_model if calibrated_model is not None else model
    probas = predict_model.predict_proba(X_input)[:, 1]
    tiers = ["High" if p >= RISK_THRESHOLDS["High"] else "Medium" if p >= RISK_THRESHOLDS["Medium"] else "Low" for p in probas]

    predictions = []
    for i in range(len(df)):
        row_dict = df.iloc[i].to_dict()
        row_dict["churn_probability"] = round(float(probas[i]), 4)
        row_dict["risk_tier"] = tiers[i]
        predictions.append(row_dict)

    return {"predictions": predictions, "count": len(predictions)}


# ──────────────────────────────────────────────────────────────
# PREDICTION HISTORY
# ──────────────────────────────────────────────────────────────

@app.get("/predictions/history")
def prediction_history():
    """Get prediction audit log."""
    try:
        if not os.path.exists(HISTORY_FILE):
            return []
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────

@app.get("/plots/{filename}")
def serve_plot(filename: str):
    """Serve plot images."""
    path = os.path.join(PLOTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Plot not found.")
    return FileResponse(path)


# ──────────────────────────────────────────────────────────────
# AI CHAT (Claude API with tool use)
# ──────────────────────────────────────────────────────────────

CHAT_TOOLS = [
    {
        "name": "predict_churn",
        "description": "Predict churn probability for a customer. Provide customer features as parameters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gender": {"type": "integer", "description": "0=Female, 1=Male"},
                "SeniorCitizen": {"type": "integer", "description": "0=No, 1=Yes"},
                "Partner": {"type": "integer", "description": "0=No, 1=Yes"},
                "Dependents": {"type": "integer", "description": "0=No, 1=Yes"},
                "tenure": {"type": "integer", "description": "Months as a customer (0-72)"},
                "PhoneService": {"type": "integer", "description": "0=No, 1=Yes"},
                "MultipleLines": {"type": "integer", "description": "0=No, 1=Yes"},
                "OnlineSecurity": {"type": "integer", "description": "0=No, 1=Yes"},
                "OnlineBackup": {"type": "integer", "description": "0=No, 1=Yes"},
                "DeviceProtection": {"type": "integer", "description": "0=No, 1=Yes"},
                "TechSupport": {"type": "integer", "description": "0=No, 1=Yes"},
                "StreamingTV": {"type": "integer", "description": "0=No, 1=Yes"},
                "StreamingMovies": {"type": "integer", "description": "0=No, 1=Yes"},
                "PaperlessBilling": {"type": "integer", "description": "0=No, 1=Yes"},
                "MonthlyCharges": {"type": "number"},
                "TotalCharges": {"type": "number"},
                "InternetService_Fiber optic": {"type": "integer", "description": "1 if Fiber optic"},
                "InternetService_No": {"type": "integer", "description": "1 if no internet"},
                "Contract_One year": {"type": "integer"},
                "Contract_Two year": {"type": "integer"},
                "PaymentMethod_Credit card (automatic)": {"type": "integer"},
                "PaymentMethod_Electronic check": {"type": "integer"},
                "PaymentMethod_Mailed check": {"type": "integer"},
            },
            "required": ["tenure", "MonthlyCharges", "TotalCharges"],
        },
    },
    {
        "name": "get_model_metrics",
        "description": "Get performance metrics (accuracy, precision, recall, F1, ROC-AUC) for all trained models.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "analyze_business_impact",
        "description": "Calculate business impact of using the churn model for proactive retention.",
        "input_schema": {
            "type": "object",
            "properties": {
                "avg_monthly_revenue": {"type": "number", "description": "Average monthly revenue per customer", "default": 65},
                "retention_cost": {"type": "number", "description": "Cost of retention offer per customer", "default": 20},
                "months_saved": {"type": "integer", "description": "Months of revenue saved per retained customer", "default": 6},
            },
        },
    },
    {
        "name": "get_churn_rates",
        "description": "Get churn rates grouped by a feature (e.g., Contract, InternetService, gender, PaymentMethod, SeniorCitizen).",
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {"type": "string", "description": "Feature to group by"},
            },
            "required": ["group_by"],
        },
    },
    {
        "name": "get_high_risk_customers",
        "description": "Get the top N customers most likely to churn, with their risk scores and top churn reasons.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of customers to return", "default": 10},
            },
        },
    },
]


def _execute_tool(name: str, args: dict) -> dict:
    """Execute a chat tool and return the result."""
    if name == "predict_churn":
        defaults = {
            "gender": 0, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
            "PhoneService": 1, "MultipleLines": 0, "OnlineSecurity": 0,
            "OnlineBackup": 0, "DeviceProtection": 0, "TechSupport": 0,
            "StreamingTV": 0, "StreamingMovies": 0, "PaperlessBilling": 0,
            "InternetService_Fiber optic": 0, "InternetService_No": 0,
            "Contract_One year": 0, "Contract_Two year": 0,
            "PaymentMethod_Credit card (automatic)": 0,
            "PaymentMethod_Electronic check": 0,
            "PaymentMethod_Mailed check": 0,
        }
        input_dict = {**defaults, **args}
        return _predict_single(input_dict)

    elif name == "get_model_metrics":
        results_path = os.path.join(MODELS_DIR, "test_results.json")
        with open(results_path) as f:
            results = json.load(f)
        return {"best_model": _state.get("best_name", "Unknown"), "results": results}

    elif name == "analyze_business_impact":
        avg_rev = args.get("avg_monthly_revenue", 65)
        ret_cost = args.get("retention_cost", 20)
        months = args.get("months_saved", 6)
        best_name = _state["best_name"]
        model = _state["models"][best_name]
        X_test = _state["X_test"]
        y_test = _state["y_test"]
        needs_scaling = {"Logistic Regression", "Neural Network"}
        if best_name in needs_scaling:
            X_eval = pd.DataFrame(_state["scaler"].transform(X_test), columns=X_test.columns, index=X_test.index)
        else:
            X_eval = X_test
        return compute_business_impact(model, X_eval, y_test, avg_rev, ret_cost, months)

    elif name == "get_churn_rates":
        group_by = args.get("group_by", "Contract")
        df = _state["df_raw"]
        if group_by not in df.columns:
            return {"error": f"Column '{group_by}' not found"}
        rates = df.groupby(group_by)["Churn"].apply(lambda x: (x == "Yes").mean())
        return {str(k): round(float(v), 4) for k, v in rates.items()}

    elif name == "get_high_risk_customers":
        limit = args.get("limit", 10)
        # Use the endpoint logic but simplified
        return high_risk_customers(limit=limit)

    return {"error": f"Unknown tool: {name}"}


@app.post("/chat")
async def chat(request: ChatRequest):
    """AI chat endpoint with Claude API and tool use. Streams SSE."""
    try:
        import anthropic
    except ImportError:
        raise HTTPException(status_code=500, detail="anthropic package not installed. Run: pip install anthropic")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set.")

    client = anthropic.Anthropic()

    system_prompt = (
        "You are an AI assistant for a Customer Churn Intelligence Platform. "
        "You help users understand customer churn patterns, make predictions, and analyze business impact. "
        "You have access to tools that can predict churn, get model metrics, analyze business impact, "
        "get churn rates by segment, and identify high-risk customers. "
        "Use these tools when users ask questions that require data. "
        "Be concise and actionable in your responses. Format numbers nicely."
    )

    messages = request.messages

    async def generate():
        nonlocal messages
        current_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        while True:
            response = client.messages.create(
                model=os.environ.get("CHAT_MODEL", "claude-sonnet-4-20250514"),
                max_tokens=1024,
                system=system_prompt,
                tools=CHAT_TOOLS,
                messages=current_messages,
            )

            # Process response content blocks
            has_tool_use = False
            tool_results = []

            for block in response.content:
                if block.type == "text":
                    # Stream text in chunks
                    text = block.text
                    chunk_size = 20
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i + chunk_size]
                        yield f"data: {json.dumps({'type': 'text_delta', 'content': chunk})}\n\n"

                elif block.type == "tool_use":
                    has_tool_use = True
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id

                    # Execute the tool
                    tool_result = _execute_tool(tool_name, tool_input)

                    # Stream tool result to frontend
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'result': tool_result})}\n\n"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": json.dumps(tool_result),
                    })

            if not has_tool_use:
                break

            # Continue conversation with tool results
            current_messages.append({"role": "assistant", "content": [{"type": "text", "text": block.text} if block.type == "text" else {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input} for block in response.content]})
            current_messages.append({"role": "user", "content": tool_results})

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ──────────────────────────────────────────────────────────────
# A/B TESTING ENDPOINTS
# ──────────────────────────────────────────────────────────────

@app.post("/experiments")
def create_experiment_endpoint(config: ExperimentCreate):
    """Create and run a new A/B test experiment."""
    try:
        result = run_experiment(
            name=config.name,
            intervention_type=config.intervention_type,
            intervention_description=config.intervention_description,
            expected_effect_size=config.expected_effect_size,
            cost_per_customer=config.cost_per_customer,
            risk_tiers=config.risk_tiers,
            feature_filters=config.feature_filters,
            split_ratio=config.split_ratio,
            significance_level=config.significance_level,
            power=config.power,
            random_seed=config.random_seed,
            avg_monthly_revenue=config.avg_monthly_revenue,
            months_saved=config.months_saved,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/experiments")
def list_experiments_endpoint():
    """List all experiments with summary statistics."""
    return get_experiments()


@app.get("/experiments/{experiment_id}")
def get_experiment_endpoint(experiment_id: str):
    """Get full experiment details including statistical results."""
    try:
        return load_experiment(experiment_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/experiments/power-analysis")
def power_analysis_endpoint(request: PowerAnalysisRequest):
    """Standalone power analysis calculator."""
    from dataclasses import asdict
    result = compute_power_analysis(
        baseline_churn_rate=request.baseline_churn_rate,
        minimum_detectable_effect=request.minimum_detectable_effect,
        alpha=request.alpha,
        power=request.power,
        eligible_population=request.eligible_population,
    )
    return asdict(result)


# ──────────────────────────────────────────────────────────────
# STATIC FILES (for production build)
# ──────────────────────────────────────────────────────────────

# Serve React frontend in production
frontend_dist = os.path.join(PROJECT_DIR, "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
