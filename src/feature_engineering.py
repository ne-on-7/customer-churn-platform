"""
Feature engineering module for Customer Churn Intelligence Platform.
Creates business-meaningful features from the encoded Telco dataset.
"""

import pandas as pd
import numpy as np


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-driven engineered features to the dataset."""
    df = df.copy()

    # Tenure bucket: groups customers by relationship length
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=[0, 1, 2, 3]  # 0=new, 1=growing, 2=established, 3=loyal
    ).astype(int)

    # Monthly-to-total spend ratio (high ratio = new or overpaying customer)
    df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # Average monthly spend (smoothed by tenure)
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Spend difference: actual vs average (positive = spending more recently)
    df["spend_trend"] = df["MonthlyCharges"] - df["avg_monthly_spend"]

    # Service count: how many add-on services the customer has
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    existing_service_cols = [c for c in service_cols if c in df.columns]
    df["services_count"] = df[existing_service_cols].sum(axis=1)

    # Has premium support: both TechSupport and OnlineSecurity
    if "TechSupport" in df.columns and "OnlineSecurity" in df.columns:
        df["has_premium_support"] = ((df["TechSupport"] == 1) & (df["OnlineSecurity"] == 1)).astype(int)

    # Contract risk score: month-to-month is highest churn risk
    contract_cols = [c for c in df.columns if c.startswith("Contract_")]
    if contract_cols:
        # If only Contract_One year and Contract_Two year exist (drop_first removed Month-to-month)
        # Month-to-month = both 0 → risk = 2
        one_year_col = [c for c in contract_cols if "One" in c]
        two_year_col = [c for c in contract_cols if "Two" in c]
        df["contract_risk_score"] = 2  # default: month-to-month
        if one_year_col:
            df.loc[df[one_year_col[0]] == 1, "contract_risk_score"] = 1
        if two_year_col:
            df.loc[df[two_year_col[0]] == 1, "contract_risk_score"] = 0

    return df


def get_feature_descriptions() -> dict:
    """Return human-readable descriptions of engineered features."""
    return {
        "tenure_bucket": "Customer tenure group (0=new <1yr, 1=growing 1-2yr, 2=established 2-4yr, 3=loyal 4-6yr)",
        "monthly_to_total_ratio": "Monthly charges relative to total spend (high = new/risky customer)",
        "avg_monthly_spend": "Average monthly spend over customer lifetime",
        "spend_trend": "Difference between current monthly charges and historical average",
        "services_count": "Number of active add-on services (0-6)",
        "has_premium_support": "Has both TechSupport and OnlineSecurity (1=yes)",
        "contract_risk_score": "Contract churn risk (0=Two year, 1=One year, 2=Month-to-month)",
    }
