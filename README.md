# Customer Churn Intelligence Platform

A full-stack machine learning platform that predicts customer churn, explains predictions using SHAP, and serves results through a REST API and interactive dashboard.

## Overview

Built on the IBM Telco Customer Churn dataset (7,043 customers), this project demonstrates end-to-end data science: from exploratory analysis through model training, explainability, and deployment.

**Key features:**
- 5 model comparison (Logistic Regression, Random Forest, XGBoost, Gradient Boosting, PyTorch Neural Network)
- 5-fold stratified cross-validation
- SHAP explainability (global + per-customer reasons)
- Business impact analysis with ROI estimation
- A/B testing experimentation layer with power analysis, outcome simulation, and statistical significance testing
- FastAPI REST API for model serving
- Streamlit dashboard with 5 interactive tabs
- Dockerized for reproducible deployment

## Project Structure

```
customer-churn-platform/
├── data/raw/                  # IBM Telco dataset
├── notebooks/01_eda.ipynb     # Exploratory data analysis
├── src/
│   ├── data_processing.py     # Load, clean, encode, split
│   ├── feature_engineering.py # Business-driven features
│   ├── models.py              # 5 model definitions (incl. PyTorch)
│   ├── train.py               # Training pipeline with CV
│   ├── evaluate.py            # Metrics, plots, business impact
│   ├── explain.py             # SHAP explanations
│   └── experimentation.py    # A/B testing & experimentation
├── api/main.py                # FastAPI REST API
├── dashboard/app.py           # Streamlit dashboard
├── Dockerfile
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
cd customer-churn-platform
pip install -r requirements.txt
```

### 2. Train all models

```bash
python -m src.train
```

This trains 5 models with 5-fold cross-validation, saves trained models to `models/`, and prints performance metrics.

### 3. Generate evaluation plots

```bash
python -m src.evaluate
```

Generates ROC curves, Precision-Recall curves, confusion matrices, and business impact analysis in `plots/`.

### 4. Generate SHAP explanations

```bash
python -m src.explain
```

### 5. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

### 6. Launch the API

```bash
uvicorn api.main:app --reload
```

API docs available at `http://localhost:8000/docs`

### Docker

```bash
docker build -t churn-platform .
docker run -p 8501:8501 churn-platform
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict churn probability + explanations |
| `GET` | `/models` | Performance summary of all models |
| `GET` | `/health` | Health check |
| `POST` | `/experiments` | Create & run an A/B test experiment |
| `GET` | `/experiments` | List all experiments |
| `GET` | `/experiments/{id}` | Get experiment details & results |
| `POST` | `/experiments/power-analysis` | Power analysis calculator |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender": 1, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0, "tenure": 2, "PhoneService": 1, "MultipleLines": 0, "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0, "PaperlessBilling": 1, "MonthlyCharges": 80.0, "TotalCharges": 160.0, "InternetService_Fiber optic": 1, "InternetService_No": 0, "Contract_One year": 0, "Contract_Two year": 0, "PaymentMethod_Credit card (automatic)": 0, "PaymentMethod_Electronic check": 1, "PaymentMethod_Mailed check": 0}'
```

## A/B Testing & Experimentation

The platform includes a full experimentation layer for testing retention interventions:

- **4 intervention types**: Discount, personalized email, service upgrade, loyalty program — each with heterogeneous treatment effects that vary by customer features
- **Power analysis**: Sample size calculator using two-proportion z-test formula
- **Statistical analysis**: Z-test, chi-squared, 95% confidence intervals, Cohen's h effect size, NNT
- **Segment breakdown**: See where interventions work best across tenure, contract type, and service count segments
- **ROI waterfall**: Revenue saved vs. intervention cost with net benefit calculation
- **Dashboard**: Interactive experiment creation, power calculator, results visualization, and history

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Models | scikit-learn, XGBoost, PyTorch |
| Explainability | SHAP |
| API | FastAPI |
| Dashboard | Streamlit, Plotly |
| Containerization | Docker |

## Author

Pratheek Annadanam
