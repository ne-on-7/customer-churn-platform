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
from src.experimentation import (
    create_experiment,
    list_experiments,
    load_experiment,
    compute_power_analysis,
)

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
tab = st.sidebar.radio("Navigate", ["Predict", "Model Comparison", "Data Explorer", "Business Impact", "A/B Testing"])

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


# ──────────────────────────────────────────────────────────────
# TAB 5: A/B TESTING
# ──────────────────────────────────────────────────────────────
elif tab == "A/B Testing" and models_loaded:
    st.title("A/B Testing & Experimentation")
    st.markdown("Design and run retention experiments on churn-prone customer segments.")

    ab_tab1, ab_tab2, ab_tab3, ab_tab4 = st.tabs([
        "Create Experiment", "Power Analysis", "Experiment Results", "Experiment History"
    ])

    # ── Sub-tab 1: Create Experiment ──
    with ab_tab1:
        st.subheader("Design a New Experiment")

        col1, col2 = st.columns(2)
        with col1:
            exp_name = st.text_input("Experiment Name", "Discount for High-Risk Customers")
            intervention_type = st.selectbox("Intervention Type", [
                "discount", "personalized_email", "service_upgrade", "loyalty_program"
            ])
            intervention_desc = st.text_input("Description", "20% monthly discount for 3 months")
            expected_effect = st.slider("Expected Effect Size (absolute churn reduction)", 0.01, 0.50, 0.15, 0.01)
            cost_per = st.number_input("Cost per Customer ($)", 0.0, 500.0, 20.0)

        with col2:
            risk_tiers = st.multiselect("Target Risk Tiers", ["High", "Medium", "Low"], default=["High"])
            split_ratio = st.slider("Treatment Group Ratio", 0.1, 0.9, 0.5, 0.05)
            sig_level = st.select_slider("Significance Level (alpha)", [0.01, 0.05, 0.10], value=0.05)
            power_level = st.select_slider("Statistical Power", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95], value=0.80)
            seed = st.number_input("Random Seed", 1, 9999, 42)
            avg_rev = st.number_input("Avg Monthly Revenue ($)", 1.0, 500.0, 65.0)
            months = st.slider("Months of Revenue Saved", 1, 24, 6)

        if st.button("Create & Run Experiment", type="primary"):
            with st.spinner("Running experiment..."):
                try:
                    result = create_experiment(
                        name=exp_name,
                        intervention_type=intervention_type,
                        intervention_description=intervention_desc,
                        expected_effect_size=expected_effect,
                        cost_per_customer=cost_per,
                        risk_tiers=risk_tiers,
                        split_ratio=split_ratio,
                        significance_level=sig_level,
                        power=power_level,
                        random_seed=int(seed),
                        avg_monthly_revenue=avg_rev,
                        months_saved=months,
                    )

                    config = result["config"]
                    res = result["results"]
                    sa = res["statistical_analysis"]
                    bi = res["business_impact"]

                    st.success(f"Experiment created: **{config['experiment_id']}**")

                    # Summary metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Control Churn Rate", f"{res['outcomes']['control']['churn_rate']:.1%}")
                    m2.metric("Treatment Churn Rate", f"{res['outcomes']['treatment']['churn_rate']:.1%}")
                    m3.metric("Absolute Lift", f"{sa['absolute_lift']:.1%}")
                    m4.metric("P-Value", f"{sa['p_value']:.4f}")

                    if sa["is_significant"]:
                        st.success(f"Result is **statistically significant** at alpha={sig_level}.")
                    else:
                        st.warning(f"Result is **not statistically significant** at alpha={sig_level}.")

                    st.markdown(f"""
                    **Business Impact:** {bi['customers_retained_by_intervention']} additional customers retained |
                    Revenue saved: ${bi['revenue_saved']:,.0f} |
                    Cost: ${bi['intervention_cost']:,.0f} |
                    Net ROI: ${bi['net_roi']:,.0f} ({bi['roi_percent']}%)
                    """)

                except Exception as e:
                    st.error(f"Failed to create experiment: {e}")

    # ── Sub-tab 2: Power Analysis Calculator ──
    with ab_tab2:
        st.subheader("Power Analysis Calculator")
        st.markdown("Determine the sample size needed to detect a given effect with statistical confidence.")

        pa_col1, pa_col2 = st.columns(2)
        with pa_col1:
            pa_baseline = st.slider("Baseline Churn Rate", 0.05, 0.95, 0.42, 0.01, key="pa_baseline")
            pa_mde = st.slider("Minimum Detectable Effect", 0.01, 0.40, 0.10, 0.01, key="pa_mde")
        with pa_col2:
            pa_alpha = st.select_slider("Significance Level", [0.01, 0.05, 0.10], value=0.05, key="pa_alpha")
            pa_power = st.select_slider("Power", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95], value=0.80, key="pa_power")

        pa_result = compute_power_analysis(pa_baseline, pa_mde, pa_alpha, pa_power)

        r1, r2, r3 = st.columns(3)
        r1.metric("Required per Group", f"{pa_result.required_sample_size_per_group:,}")
        r2.metric("Total Required", f"{pa_result.total_required:,}")
        r3.metric("Power", f"{pa_result.achieved_power:.0%}")

        # Sample size vs effect size curve
        st.subheader("Sample Size vs. Effect Size")
        effect_sizes = np.arange(0.02, 0.41, 0.01)
        sample_sizes = []
        for es in effect_sizes:
            pa_temp = compute_power_analysis(pa_baseline, es, pa_alpha, pa_power)
            sample_sizes.append(pa_temp.required_sample_size_per_group)

        fig_pa = go.Figure()
        fig_pa.add_trace(go.Scatter(
            x=effect_sizes, y=sample_sizes,
            mode="lines", name="Required n per group",
            line=dict(color="#3498db", width=3),
        ))
        fig_pa.add_vline(x=pa_mde, line_dash="dash", line_color="red",
                         annotation_text=f"Selected MDE = {pa_mde:.2f}")
        fig_pa.update_layout(
            xaxis_title="Minimum Detectable Effect (absolute)",
            yaxis_title="Required Sample Size per Group",
            title="How Effect Size Impacts Required Sample Size",
            height=400,
        )
        st.plotly_chart(fig_pa, use_container_width=True)

    # ── Sub-tab 3: Experiment Results ──
    with ab_tab3:
        st.subheader("Experiment Results")

        experiments = list_experiments()
        if not experiments:
            st.info("No experiments yet. Create one in the 'Create Experiment' tab.")
        else:
            exp_names = {e["experiment_id"]: f"{e['name']} ({e['experiment_id']})" for e in experiments}
            selected_id = st.selectbox("Select Experiment", list(exp_names.keys()),
                                       format_func=lambda x: exp_names[x])

            exp_data = load_experiment(selected_id)
            config = exp_data["config"]
            results = exp_data.get("results", {})

            if not results:
                st.warning("This experiment has no results yet.")
            else:
                outcomes = results["outcomes"]
                sa = results["statistical_analysis"]
                bi = results["business_impact"]

                # Significance banner
                if sa["is_significant"]:
                    st.success(
                        f"**Statistically Significant** (p = {sa['p_value']:.4f} < "
                        f"alpha = {config['design']['significance_level']}). "
                        f"The intervention reduced churn by {abs(sa['absolute_lift']):.1%} "
                        f"(relative: {abs(sa['relative_lift']):.1%})."
                    )
                else:
                    st.warning(
                        f"**Not Statistically Significant** (p = {sa['p_value']:.4f} >= "
                        f"alpha = {config['design']['significance_level']}). "
                        f"Cannot conclude the intervention had an effect."
                    )

                # Key metrics row
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Absolute Lift", f"{sa['absolute_lift']:.1%}")
                m2.metric("P-Value", f"{sa['p_value']:.4f}")
                m3.metric("95% CI", f"[{sa['confidence_interval_95'][0]:.1%}, {sa['confidence_interval_95'][1]:.1%}]")
                m4.metric("NNT", f"{sa['number_needed_to_treat']:.1f}")
                m5.metric("Cohen's h", f"{sa['effect_size_cohens_h']:.3f}")

                st.markdown("---")

                # Chart 1: Churn rate comparison with CI
                st.subheader("Churn Rate: Control vs Treatment")
                ci = sa["confidence_interval_95"]
                fig_compare = go.Figure()

                ctrl_rate = outcomes["control"]["churn_rate"]
                treat_rate = outcomes["treatment"]["churn_rate"]

                fig_compare.add_trace(go.Bar(
                    x=["Control", "Treatment"],
                    y=[ctrl_rate, treat_rate],
                    marker_color=["#e74c3c", "#2ecc71"],
                    text=[f"{ctrl_rate:.1%}", f"{treat_rate:.1%}"],
                    textposition="outside",
                    error_y=dict(
                        type="data",
                        array=[
                            1.96 * np.sqrt(ctrl_rate * (1 - ctrl_rate) / outcomes["control"]["n"]),
                            1.96 * np.sqrt(treat_rate * (1 - treat_rate) / outcomes["treatment"]["n"]),
                        ],
                        visible=True,
                    ),
                ))
                fig_compare.update_layout(
                    yaxis_title="Churn Rate",
                    yaxis_tickformat=".0%",
                    title=f"Intervention: {config['intervention']['type'].replace('_', ' ').title()}",
                    height=400,
                    showlegend=False,
                )
                st.plotly_chart(fig_compare, use_container_width=True)

                # Chart 2: Segment breakdown
                segments = results.get("segment_breakdown", [])
                if segments:
                    st.subheader("Segment Breakdown")

                    seg_df = pd.DataFrame(segments)
                    fig_seg = go.Figure()
                    fig_seg.add_trace(go.Bar(
                        name="Control",
                        x=seg_df["segment"],
                        y=seg_df["control_churn_rate"],
                        marker_color="#e74c3c",
                    ))
                    fig_seg.add_trace(go.Bar(
                        name="Treatment",
                        x=seg_df["segment"],
                        y=seg_df["treatment_churn_rate"],
                        marker_color="#2ecc71",
                    ))
                    fig_seg.update_layout(
                        barmode="group",
                        yaxis_title="Churn Rate",
                        yaxis_tickformat=".0%",
                        title="Churn Rate by Segment: Control vs Treatment",
                        height=400,
                    )
                    st.plotly_chart(fig_seg, use_container_width=True)

                # Chart 3: ROI Waterfall
                st.subheader("Business Impact (ROI Waterfall)")
                fig_waterfall = go.Figure(go.Waterfall(
                    name="ROI",
                    orientation="v",
                    x=["Revenue Saved", "Intervention Cost", "Net ROI"],
                    y=[bi["revenue_saved"], -bi["intervention_cost"], bi["net_roi"]],
                    measure=["relative", "relative", "total"],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    text=[f"${bi['revenue_saved']:,.0f}",
                          f"-${bi['intervention_cost']:,.0f}",
                          f"${bi['net_roi']:,.0f}"],
                    textposition="outside",
                    decreasing={"marker": {"color": "#e74c3c"}},
                    increasing={"marker": {"color": "#2ecc71"}},
                    totals={"marker": {"color": "#3498db"}},
                ))
                fig_waterfall.update_layout(
                    title=f"Experiment ROI: {bi['roi_percent']}%",
                    yaxis_title="Amount ($)",
                    height=400,
                    showlegend=False,
                )
                st.plotly_chart(fig_waterfall, use_container_width=True)

    # ── Sub-tab 4: Experiment History ──
    with ab_tab4:
        st.subheader("Experiment History")

        experiments = list_experiments()
        if not experiments:
            st.info("No experiments yet. Create one in the 'Create Experiment' tab.")
        else:
            hist_data = []
            for e in experiments:
                row = {
                    "Name": e["name"],
                    "Status": e["status"],
                    "Intervention": e.get("intervention_type", ""),
                    "Risk Tiers": ", ".join(e.get("risk_tiers", [])),
                    "Sample Size": e.get("sample_size", "—"),
                    "Lift": f"{e['absolute_lift']:.1%}" if e.get("absolute_lift") is not None else "—",
                    "P-Value": f"{e['p_value']:.4f}" if e.get("p_value") is not None else "—",
                    "Significant": "Yes" if e.get("is_significant") else "No",
                    "ROI %": f"{e['roi_percent']}%" if e.get("roi_percent") is not None else "—",
                }
                hist_data.append(row)

            hist_df = pd.DataFrame(hist_data)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
