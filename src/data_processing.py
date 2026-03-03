"""
Data processing module for Customer Churn Intelligence Platform.
Handles loading, cleaning, encoding, and splitting the Telco dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_PATH = os.path.join(DATA_DIR, "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load raw CSV and perform initial cleaning."""
    df = pd.read_csv(path)

    # TotalCharges has whitespace strings for new customers (tenure=0) → convert
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill NaN TotalCharges with 0 (these are brand-new customers)
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Drop customerID — not a feature
    df = df.drop(columns=["customerID"])

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables properly."""
    df = df.copy()

    # Binary columns → 0/1
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
                   "PaperlessBilling", "Churn"]
    for col in binary_cols:
        df[col] = df[col].map(binary_map).astype(int)

    # Multi-value binary columns (have "No internet service" / "No phone service")
    multi_binary_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                         "DeviceProtection", "TechSupport", "StreamingTV",
                         "StreamingMovies"]
    for col in multi_binary_cols:
        df[col] = df[col].apply(lambda x: 1 if x == "Yes" else 0)

    # OneHotEncode nominal categoricals
    nominal_cols = ["InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test with stratification on Churn."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def get_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit and return a StandardScaler on training data."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def run_pipeline(save: bool = True):
    """Run full data processing pipeline and optionally save artifacts."""
    print("[1/4] Loading raw data...")
    df = load_raw_data()
    print(f"      Loaded {len(df)} rows, {len(df.columns)} columns")

    print("[2/4] Encoding features...")
    df_encoded = encode_features(df)
    print(f"      Encoded shape: {df_encoded.shape}")

    print("[3/4] Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = split_data(df_encoded)
    print(f"      Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"      Churn rate — Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

    print("[4/4] Fitting scaler...")
    scaler = get_scaler(X_train)

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        processed_dir = os.path.join(DATA_DIR, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

        # Save feature names for API/dashboard use
        joblib.dump(list(X_train.columns), os.path.join(MODELS_DIR, "feature_names.pkl"))

        print(f"\n      Saved processed data to {processed_dir}")
        print(f"      Saved scaler to {MODELS_DIR}/scaler.pkl")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    run_pipeline(save=True)
