"""
Streamlit Dashboard for Customer Churn Intelligence Platform.
Interactive UI for predictions, model comparison, data exploration, and business impact.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
import shap

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_DIR)

from src.data_processing import load_raw_data, encode_features
from src.feature_engineering import add_engineered_features
from src.evaluate import load_trained_models
from src.explain import local_explanation, get_top_reasons

MODELS_DIR = os.path.join(PROJECT_DIR, "models")

st.set_page_config(page_title="Customer Churn Intelligence", page_icon="📊", layout="wide")


@st.cache_resource
def load_models():
    """Load all trained models and artifacts."""
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    best_name = open(os.path.join(MODELS_DIR, "best_model.txt")).read().strip()
    models = load_trained_models(input_dim=len(feature_names))
    return feature_names, scaler, best_name, models


@st.cache_data
def load_data():
    """Load raw and processed data."""
    df_raw = load_raw_data()
    processed_dir = os.path.join(PROJECT_DIR, "data", "processed")
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    return df_raw, X_train, X_test, y_test


# Sidebar
st.sidebar.title("Customer Churn Intelligence")
st.sidebar.markdown("---")
tab = st.sidebar.radio("Navigate", ["Predict", "Model Comparison", "Data Explorer", "Business Impact"])

try:
    feature_names, scaler, best_name, models = load_models()
    df_raw, X_train, X_test, y_test = load_data()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Models not loaded. Run `python -m src.train` first.\n\nError: {e}")


# ──────────────────────────────────────────────────────────────
# TAB 1: PREDICT
# ──────────────────────────────────────────────────────────────
if tab == "Predict" and models_loaded:
    st.title("Churn Risk Prediction")
    st.markdown("Enter customer details to predict their churn probability and see what's driving the risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.subheader("Services")
        phone = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["No", "Yes"])
        backup = st.selectbox("Online Backup", ["No", "Yes"])
        protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech = st.selectbox("Tech Support", ["No", "Yes"])
        tv = st.selectbox("Streaming TV", ["No", "Yes"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes"])

    with col3:
        st.subheader("Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly * tenure)

    if st.button("Predict Churn Risk", type="primary", width="stretch"):
        # Build feature dict
        yn = lambda x: 1 if x == "Yes" else 0
        input_data = {
            "gender": 1 if gender == "Male" else 0,
            "SeniorCitizen": yn(senior),
            "Partner": yn(partner),
            "Dependents": yn(dependents),
            "tenure": tenure,
            "PhoneService": yn(phone),
            "MultipleLines": yn(multiple_lines),
            "OnlineSecurity": yn(security),
            "OnlineBackup": yn(backup),
            "DeviceProtection": yn(protection),
            "TechSupport": yn(tech),
            "StreamingTV": yn(tv),
            "StreamingMovies": yn(movies),
            "PaperlessBilling": yn(paperless),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
            "InternetService_No": 1 if internet == "No" else 0,
            "Contract_One year": 1 if contract == "One year" else 0,
            "Contract_Two year": 1 if contract == "Two year" else 0,
            "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
            "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
        }

        df_input = pd.DataFrame([input_data])
        df_input = add_engineered_features(df_input)

        for col in feature_names:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[feature_names]

        model = models[best_name]
        needs_scaling = {"Logistic Regression", "Neural Network"}
        if best_name in needs_scaling:
            X_pred = pd.DataFrame(scaler.transform(df_input), columns=df_input.columns)
        else:
            X_pred = df_input

        proba = model.predict_proba(X_pred)[0, 1]

        if proba >= 0.7:
            risk_tier, color = "HIGH RISK", "🔴"
        elif proba >= 0.4:
            risk_tier, color = "MEDIUM RISK", "🟡"
        else:
            risk_tier, color = "LOW RISK", "🟢"

        st.markdown("---")
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Churn Probability", f"{proba:.1%}")
            st.markdown(f"### {color} {risk_tier}")
            st.caption(f"Model: {best_name}")

        with r2:
            try:
                sv, feat_names = local_explanation(model, best_name, X_train, df_input)
                reasons = get_top_reasons(sv, feat_names, top_n=5)
                st.markdown("### Top Churn Drivers")
                for i, reason in enumerate(reasons, 1):
                    st.markdown(f"**{i}.** {reason}")
            except Exception as ex:
                st.warning(f"Could not generate explanation: {ex}")


# ──────────────────────────────────────────────────────────────
# TAB 2: MODEL COMPARISON
# ──────────────────────────────────────────────────────────────
elif tab == "Model Comparison" and models_loaded:
    st.title("Model Performance Comparison")

    results_path = os.path.join(MODELS_DIR, "test_results.json")
    cv_path = os.path.join(MODELS_DIR, "cv_results.json")

    if os.path.exists(results_path):
        with open(results_path) as f:
            test_results = json.load(f)

        # Metrics table
        df_results = pd.DataFrame(test_results).T
        df_results.index.name = "Model"
        df_results = df_results.round(4)

        st.subheader("Test Set Metrics")
        st.dataframe(df_results.style.highlight_max(axis=0, color="#2ecc71"), width="stretch")

        # Bar chart comparison
        st.subheader("Visual Comparison")
        metric = st.selectbox("Select Metric", ["roc_auc", "f1", "precision", "recall", "accuracy"])

        fig = px.bar(
            x=list(test_results.keys()),
            y=[test_results[m][metric] for m in test_results],
            color=list(test_results.keys()),
            title=f"{metric.upper()} by Model",
            labels={"x": "Model", "y": metric},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

        st.markdown(f"**Best model by ROC-AUC:** {best_name}")

    # Show saved plots if available
    plots_dir = os.path.join(PROJECT_DIR, "plots")
    for plot_name, title in [("roc_curves.png", "ROC Curves"), ("pr_curves.png", "Precision-Recall Curves")]:
        path = os.path.join(plots_dir, plot_name)
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, width="stretch")


# ──────────────────────────────────────────────────────────────
# TAB 3: DATA EXPLORER
# ──────────────────────────────────────────────────────────────
elif tab == "Data Explorer" and models_loaded:
    st.title("Data Explorer")

    st.subheader("Dataset Overview")
    st.markdown(f"**Rows:** {len(df_raw):,} | **Features:** {len(df_raw.columns)}")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        contract_filter = st.multiselect("Contract Type", df_raw["Contract"].unique(), default=df_raw["Contract"].unique())
    with col2:
        tenure_range = st.slider("Tenure Range (months)", 0, 72, (0, 72))

    filtered = df_raw[
        (df_raw["Contract"].isin(contract_filter)) &
        (df_raw["tenure"] >= tenure_range[0]) &
        (df_raw["tenure"] <= tenure_range[1])
    ]

    st.markdown(f"**Filtered:** {len(filtered):,} customers")

    # Churn rate by group
    st.subheader("Churn Rate Analysis")
    group_by = st.selectbox("Group by", ["Contract", "InternetService", "gender", "PaymentMethod", "SeniorCitizen"])

    churn_rates = filtered.groupby(group_by)["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
    churn_rates.columns = [group_by, "Churn Rate (%)"]

    fig = px.bar(churn_rates, x=group_by, y="Churn Rate (%)", color=group_by,
                 title=f"Churn Rate by {group_by}")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    # Distribution plot
    st.subheader("Feature Distribution")
    num_col = st.selectbox("Numerical Feature", ["MonthlyCharges", "TotalCharges", "tenure"])
    fig = px.histogram(filtered, x=num_col, color="Churn", barmode="overlay", nbins=40,
                       title=f"{num_col} Distribution by Churn", opacity=0.7)
    st.plotly_chart(fig, width="stretch")


# ──────────────────────────────────────────────────────────────
# TAB 4: BUSINESS IMPACT
# ──────────────────────────────────────────────────────────────
elif tab == "Business Impact" and models_loaded:
    st.title("Business Impact Calculator")
    st.markdown("Estimate the financial impact of using this model for proactive customer retention.")

    col1, col2 = st.columns(2)
    with col1:
        avg_revenue = st.number_input("Avg Monthly Revenue per Customer ($)", 10.0, 500.0, 65.0)
        retention_cost = st.number_input("Cost of Retention Offer ($)", 0.0, 200.0, 20.0)
    with col2:
        months_saved = st.slider("Months of Revenue Saved per Retained Customer", 1, 24, 6)

    impact_path = os.path.join(MODELS_DIR, "business_impact.json")

    # Recompute based on user inputs
    model = models[best_name]
    needs_scaling = {"Logistic Regression", "Neural Network"}
    if best_name in needs_scaling:
        X_eval = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    else:
        X_eval = X_test

    from src.evaluate import compute_business_impact
    impact = compute_business_impact(model, X_eval, y_test, avg_revenue, retention_cost, months_saved)

    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Revenue Saved", f"${impact['revenue_saved']:,.0f}")
    m2.metric("Campaign Cost", f"${impact['retention_spend']:,.0f}")
    m3.metric("Net Benefit", f"${impact['net_benefit']:,.0f}")
    m4.metric("ROI", f"{impact['roi_percent']}%")

    st.markdown("---")
    st.subheader("Confusion Matrix Breakdown")

    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.markdown(f"""
        | | Predicted Retained | Predicted Churned |
        |---|---|---|
        | **Actually Retained** | {impact['true_negatives']} (correct) | {impact['false_positives']} (unnecessary offer) |
        | **Actually Churned** | {impact['false_negatives']} (missed!) | {impact['true_positives']} (caught!) |
        """)
    with cm_col2:
        st.markdown(f"""
        **Interpretation:**
        - **{impact['true_positives']}** churning customers caught → saved **${impact['revenue_saved']:,.0f}**
        - **{impact['false_negatives']}** churning customers missed → lost **${impact['missed_revenue']:,.0f}**
        - **{impact['false_positives']}** unnecessary offers sent → wasted **${impact['false_positives'] * retention_cost:,.0f}**
        """)
