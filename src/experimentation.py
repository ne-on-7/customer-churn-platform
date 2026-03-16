"""
A/B Testing & Experimentation module for Customer Churn Intelligence Platform.
Supports experiment creation, power analysis, outcome simulation, and statistical analysis.
"""

import os
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from src.feature_engineering import add_engineered_features
from src.evaluate import load_trained_models

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR, "data", "experiments")

RISK_THRESHOLDS = {"High": 0.5, "Medium": 0.2, "Low": 0.0}

INTERVENTION_PROFILES = {
    "discount": {
        "base_reduction": 0.15,
        "modifier_feature": "MonthlyCharges",
        "modifier_direction": "high",  # higher value = stronger effect
        "noise_std": 0.03,
    },
    "personalized_email": {
        "base_reduction": 0.08,
        "modifier_feature": "tenure",
        "modifier_direction": "mid",  # mid-range = strongest
        "noise_std": 0.04,
    },
    "service_upgrade": {
        "base_reduction": 0.12,
        "modifier_feature": "services_count",
        "modifier_direction": "low",  # fewer services = more to gain
        "noise_std": 0.03,
    },
    "loyalty_program": {
        "base_reduction": 0.10,
        "modifier_feature": "tenure",
        "modifier_direction": "high",  # longer tenure = values loyalty
        "noise_std": 0.035,
    },
}


@dataclass
class PowerAnalysisResult:
    baseline_churn_rate: float
    minimum_detectable_effect: float
    required_sample_size_per_group: int
    total_required: int
    eligible_population: int
    is_feasible: bool
    achieved_power: float


def compute_power_analysis(
    baseline_churn_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
    eligible_population: int = 0,
) -> PowerAnalysisResult:
    """Compute minimum sample size per group for a two-proportion z-test."""
    p1 = baseline_churn_rate
    p2 = max(0.01, p1 - minimum_detectable_effect)

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p_bar = (p1 + p2) / 2
    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p1 - p2) ** 2

    n_per_group = int(np.ceil(numerator / denominator))
    total_required = n_per_group * 2
    is_feasible = eligible_population >= total_required if eligible_population > 0 else True

    return PowerAnalysisResult(
        baseline_churn_rate=round(p1, 4),
        minimum_detectable_effect=round(minimum_detectable_effect, 4),
        required_sample_size_per_group=n_per_group,
        total_required=total_required,
        eligible_population=eligible_population,
        is_feasible=is_feasible,
        achieved_power=round(power, 4),
    )


def get_eligible_customers(
    X_data: pd.DataFrame,
    y_proba: np.ndarray,
    risk_tiers: list = None,
    feature_filters: dict = None,
) -> pd.DataFrame:
    """Filter customers by risk tier and feature criteria."""
    df = X_data.copy()
    df["churn_probability"] = y_proba

    # Assign risk tiers using shared thresholds
    df["risk_tier"] = "Low"
    df.loc[df["churn_probability"] >= RISK_THRESHOLDS["Medium"], "risk_tier"] = "Medium"
    df.loc[df["churn_probability"] >= RISK_THRESHOLDS["High"], "risk_tier"] = "High"

    # Filter by risk tier
    if risk_tiers:
        df = df[df["risk_tier"].isin(risk_tiers)]

    # Filter by feature values
    if feature_filters:
        for col, val in feature_filters.items():
            if col in df.columns:
                df = df[df[col] == val]

    return df.reset_index(drop=True)


def assign_customers(
    eligible_df: pd.DataFrame,
    split_ratio: float = 0.5,
    stratify_by: list = None,
    random_seed: int = 42,
) -> tuple:
    """Randomly assign eligible customers to treatment vs control groups."""
    if len(eligible_df) < 4:
        # Too few for stratified split
        treatment = eligible_df.sample(frac=split_ratio, random_state=random_seed)
        control = eligible_df.drop(treatment.index)
        return treatment.reset_index(drop=True), control.reset_index(drop=True)

    stratify_col = None
    if stratify_by:
        # Combine stratification columns into a single key
        valid_cols = [c for c in stratify_by if c in eligible_df.columns]
        if valid_cols:
            stratify_col = eligible_df[valid_cols].astype(str).agg("_".join, axis=1)
            # Ensure each stratum has at least 2 samples
            counts = stratify_col.value_counts()
            if (counts < 2).any():
                stratify_col = None

    control, treatment = train_test_split(
        eligible_df,
        test_size=split_ratio,
        stratify=stratify_col,
        random_state=random_seed,
    )
    return treatment.reset_index(drop=True), control.reset_index(drop=True)


def _compute_modifier_weights(series: pd.Series, direction: str) -> np.ndarray:
    """Compute per-customer modifier weights (0.5 to 1.5) from a feature."""
    if series.nunique() <= 1:
        return np.ones(len(series))

    # Normalize to 0-1 percentile rank
    ranks = series.rank(pct=True).values

    if direction == "high":
        weights = 0.5 + ranks  # higher value = stronger effect (0.5 to 1.5)
    elif direction == "low":
        weights = 1.5 - ranks  # lower value = stronger effect (0.5 to 1.5)
    elif direction == "mid":
        # Peak at the middle, taper at extremes
        weights = 1.5 - 2 * np.abs(ranks - 0.5)  # peaks at 1.5 when rank=0.5
    else:
        weights = np.ones(len(series))

    return np.clip(weights, 0.5, 1.5)


def simulate_experiment_outcomes(
    control_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    intervention_type: str,
    expected_effect_size: float,
    random_seed: int = 42,
) -> dict:
    """Simulate churn outcomes with heterogeneous treatment effects."""
    rng = np.random.RandomState(random_seed)

    # Control: each customer churns with their predicted probability
    control_probs = control_df["churn_probability"].values
    control_churned = rng.binomial(1, control_probs)

    # Treatment: reduce churn probability based on intervention profile
    treatment_probs = treatment_df["churn_probability"].values.copy()
    profile = INTERVENTION_PROFILES.get(intervention_type, INTERVENTION_PROFILES["discount"])

    modifier_col = profile["modifier_feature"]
    if modifier_col in treatment_df.columns:
        weights = _compute_modifier_weights(
            treatment_df[modifier_col], profile["modifier_direction"]
        )
    else:
        weights = np.ones(len(treatment_df))

    noise = rng.normal(0, profile["noise_std"], len(treatment_df))
    reduction = expected_effect_size * weights + noise
    adjusted_probs = np.clip(treatment_probs - reduction, 0.01, 0.99)
    treatment_churned = rng.binomial(1, adjusted_probs)

    return {
        "control": {
            "n": int(len(control_df)),
            "churned": int(control_churned.sum()),
            "retained": int(len(control_df) - control_churned.sum()),
            "churn_rate": round(float(control_churned.mean()), 4),
            "outcomes": control_churned.tolist(),
        },
        "treatment": {
            "n": int(len(treatment_df)),
            "churned": int(treatment_churned.sum()),
            "retained": int(len(treatment_df) - treatment_churned.sum()),
            "churn_rate": round(float(treatment_churned.mean()), 4),
            "outcomes": treatment_churned.tolist(),
        },
    }


def analyze_experiment(
    outcomes: dict,
    alpha: float = 0.05,
    cost_per_customer: float = 0.0,
    avg_monthly_revenue: float = 65.0,
    months_saved: int = 6,
) -> dict:
    """Perform full statistical analysis on experiment outcomes."""
    n_c = outcomes["control"]["n"]
    n_t = outcomes["treatment"]["n"]
    x_c = outcomes["control"]["churned"]
    x_t = outcomes["treatment"]["churned"]
    p_c = outcomes["control"]["churn_rate"]
    p_t = outcomes["treatment"]["churn_rate"]

    # Two-proportion z-test
    p_pooled = (x_c + x_t) / (n_c + n_t)
    se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_c + 1 / n_t))
    z_stat = (p_t - p_c) / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Confidence interval for difference in proportions
    se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    ci_lower = round(diff - z_crit * se_diff, 4)
    ci_upper = round(diff + z_crit * se_diff, 4)

    # Chi-squared test
    contingency = np.array([[x_c, n_c - x_c], [x_t, n_t - x_t]])
    chi2, chi2_p, _, _ = stats.chi2_contingency(contingency, correction=False)

    # Effect sizes
    cohens_h = 2 * np.arcsin(np.sqrt(p_t)) - 2 * np.arcsin(np.sqrt(p_c))
    absolute_lift = round(p_t - p_c, 4)
    relative_lift = round((p_t - p_c) / p_c, 4) if p_c > 0 else 0.0
    nnt = round(1 / abs(p_c - p_t), 2) if abs(p_c - p_t) > 0 else float("inf")

    # Business impact
    customers_retained = max(0, outcomes["control"]["churned"] - outcomes["treatment"]["churned"])
    revenue_saved = round(customers_retained * avg_monthly_revenue * months_saved, 2)
    intervention_cost = round(n_t * cost_per_customer, 2)
    net_roi = round(revenue_saved - intervention_cost, 2)
    roi_percent = round((net_roi / intervention_cost) * 100, 1) if intervention_cost > 0 else 0.0

    return {
        "statistical_analysis": {
            "absolute_lift": absolute_lift,
            "relative_lift": relative_lift,
            "z_statistic": round(z_stat, 4),
            "p_value": round(p_value, 6),
            "confidence_interval_95": [ci_lower, ci_upper],
            "is_significant": bool(p_value < alpha),
            "chi_squared": round(chi2, 4),
            "chi_squared_p_value": round(chi2_p, 6),
            "effect_size_cohens_h": round(cohens_h, 4),
            "number_needed_to_treat": nnt,
        },
        "business_impact": {
            "customers_retained_by_intervention": customers_retained,
            "revenue_saved": revenue_saved,
            "intervention_cost": intervention_cost,
            "net_roi": net_roi,
            "roi_percent": roi_percent,
        },
    }


def compute_segment_breakdown(
    control_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    control_outcomes: list,
    treatment_outcomes: list,
    segment_columns: list = None,
) -> list:
    """Break down experiment results by customer segments."""
    if segment_columns is None:
        segment_columns = ["tenure_bucket", "contract_risk_score"]

    valid_cols = [c for c in segment_columns if c in control_df.columns and c in treatment_df.columns]
    if not valid_cols:
        return []

    results = []
    ctrl = control_df.copy()
    ctrl["_churned"] = control_outcomes
    treat = treatment_df.copy()
    treat["_churned"] = treatment_outcomes

    for col in valid_cols:
        for val in sorted(set(ctrl[col].unique()) | set(treat[col].unique())):
            ctrl_seg = ctrl[ctrl[col] == val]
            treat_seg = treat[treat[col] == val]
            if len(ctrl_seg) < 2 or len(treat_seg) < 2:
                continue
            ctrl_rate = ctrl_seg["_churned"].mean()
            treat_rate = treat_seg["_churned"].mean()
            lift = round((treat_rate - ctrl_rate) / ctrl_rate, 4) if ctrl_rate > 0 else 0.0
            results.append({
                "segment": f"{col}={val}",
                "control_n": int(len(ctrl_seg)),
                "treatment_n": int(len(treat_seg)),
                "control_churn_rate": round(float(ctrl_rate), 4),
                "treatment_churn_rate": round(float(treat_rate), 4),
                "relative_lift": lift,
            })

    return results


def _load_data_and_predictions():
    """Load processed data and generate churn predictions using the best model."""
    import joblib

    processed_dir = os.path.join(PROJECT_DIR, "data", "processed")
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))

    # Combine train+test for a larger pool of customers
    X_all = pd.concat([X_train, X_test], ignore_index=True)

    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    best_name = open(os.path.join(MODELS_DIR, "best_model.txt")).read().strip()
    models = load_trained_models(input_dim=len(feature_names))
    model = models[best_name]

    # Use calibrated model if available for consistent probabilities with API
    safe_best = best_name.lower().replace(" ", "_")
    calibrated_path = os.path.join(MODELS_DIR, f"{safe_best}_calibrated.pkl")
    if os.path.exists(calibrated_path):
        model = joblib.load(calibrated_path)

    # Ensure features match
    for col in feature_names:
        if col not in X_all.columns:
            X_all[col] = 0
    X_features = X_all[feature_names]

    needs_scaling = {"Logistic Regression", "Neural Network"}
    if best_name in needs_scaling:
        X_scaled = pd.DataFrame(scaler.transform(X_features), columns=feature_names)
    else:
        X_scaled = X_features

    y_proba = model.predict_proba(X_scaled)[:, 1]

    return X_all, y_proba


def create_experiment(
    name: str,
    intervention_type: str,
    intervention_description: str = "",
    expected_effect_size: float = 0.15,
    cost_per_customer: float = 20.0,
    risk_tiers: list = None,
    feature_filters: dict = None,
    split_ratio: float = 0.5,
    stratify_by: list = None,
    significance_level: float = 0.05,
    power: float = 0.80,
    random_seed: int = 42,
    avg_monthly_revenue: float = 65.0,
    months_saved: int = 6,
) -> dict:
    """Create, run, and save a complete A/B test experiment."""
    if risk_tiers is None:
        risk_tiers = ["High"]
    if stratify_by is None:
        stratify_by = ["risk_tier", "tenure_bucket"]

    # Load data and predictions
    X_all, y_proba = _load_data_and_predictions()

    # Get eligible customers
    eligible = get_eligible_customers(X_all, y_proba, risk_tiers, feature_filters)

    if len(eligible) < 4:
        raise ValueError(f"Only {len(eligible)} eligible customers found. Need at least 4.")

    # Compute baseline churn rate
    baseline_churn = float(eligible["churn_probability"].mean())

    # Power analysis
    pa = compute_power_analysis(
        baseline_churn_rate=baseline_churn,
        minimum_detectable_effect=expected_effect_size,
        alpha=significance_level,
        power=power,
        eligible_population=len(eligible),
    )

    # Generate experiment ID
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Assign customers
    treatment_df, control_df = assign_customers(
        eligible, split_ratio=split_ratio, stratify_by=stratify_by, random_seed=random_seed
    )

    # Simulate outcomes
    outcomes = simulate_experiment_outcomes(
        control_df, treatment_df, intervention_type, expected_effect_size, random_seed
    )

    # Statistical analysis
    analysis = analyze_experiment(
        outcomes,
        alpha=significance_level,
        cost_per_customer=cost_per_customer,
        avg_monthly_revenue=avg_monthly_revenue,
        months_saved=months_saved,
    )

    # Segment breakdown
    segments = compute_segment_breakdown(
        control_df, treatment_df,
        outcomes["control"]["outcomes"],
        outcomes["treatment"]["outcomes"],
    )

    # Build config
    config = {
        "experiment_id": experiment_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "status": "completed",
        "intervention": {
            "type": intervention_type,
            "description": intervention_description,
            "expected_effect_size": expected_effect_size,
            "cost_per_customer": cost_per_customer,
        },
        "targeting": {
            "risk_tiers": risk_tiers,
            "filters": feature_filters,
        },
        "design": {
            "split_ratio": split_ratio,
            "stratify_by": stratify_by,
            "random_seed": random_seed,
            "significance_level": significance_level,
            "power": power,
        },
        "power_analysis": asdict(pa),
    }

    # Build results
    results = {
        "experiment_id": experiment_id,
        "completed_at": datetime.now().isoformat(),
        "outcomes": {
            "control": {k: v for k, v in outcomes["control"].items() if k != "outcomes"},
            "treatment": {k: v for k, v in outcomes["treatment"].items() if k != "outcomes"},
        },
        **analysis,
        "segment_breakdown": segments,
    }

    # Build assignments summary (exclude per-customer outcomes for storage)
    assignments = {
        "experiment_id": experiment_id,
        "assigned_at": datetime.now().isoformat(),
        "control_count": len(control_df),
        "treatment_count": len(treatment_df),
    }

    # Save to disk
    exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(exp_dir, "assignments.json"), "w") as f:
        json.dump(assignments, f, indent=2)

    return {
        "experiment_id": experiment_id,
        "config": config,
        "results": results,
    }


def list_experiments() -> list:
    """List all experiments with summary info."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    experiments = []

    for exp_id in sorted(os.listdir(EXPERIMENTS_DIR), reverse=True):
        config_path = os.path.join(EXPERIMENTS_DIR, exp_id, "config.json")
        results_path = os.path.join(EXPERIMENTS_DIR, exp_id, "results.json")
        if not os.path.isfile(config_path):
            continue

        with open(config_path) as f:
            config = json.load(f)

        summary = {
            "experiment_id": config["experiment_id"],
            "name": config["name"],
            "status": config["status"],
            "created_at": config["created_at"],
            "intervention_type": config["intervention"]["type"],
            "risk_tiers": config["targeting"]["risk_tiers"],
        }

        if os.path.isfile(results_path):
            with open(results_path) as f:
                results = json.load(f)
            sa = results.get("statistical_analysis", {})
            bi = results.get("business_impact", {})
            summary.update({
                "sample_size": results["outcomes"]["control"]["n"] + results["outcomes"]["treatment"]["n"],
                "absolute_lift": sa.get("absolute_lift"),
                "p_value": sa.get("p_value"),
                "is_significant": sa.get("is_significant"),
                "roi_percent": bi.get("roi_percent"),
            })

        experiments.append(summary)

    return experiments


def load_experiment(experiment_id: str) -> dict:
    """Load full experiment data."""
    exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment '{experiment_id}' not found.")

    data = {}
    for filename in ["config.json", "results.json", "assignments.json"]:
        path = os.path.join(exp_dir, filename)
        if os.path.isfile(path):
            with open(path) as f:
                data[filename.replace(".json", "")] = json.load(f)

    return data
