"""
Data Preprocessing Module for EAPS
Loads HR datasets, cleans, encodes, and prepares data for ML training.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


# Columns to drop (constant or irrelevant)
DROP_COLUMNS = [
    "EmpID", "EmployeeCount", "EmployeeNumber", "Over18",
    "StandardHours", "AgeGroup", "SalarySlab"
]

# Target column
TARGET = "Attrition"

# Columns that need label encoding
CATEGORICAL_COLUMNS = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime"
]


def load_data(data_dir: str) -> pd.DataFrame:
    """Load and combine HR datasets."""
    file1 = os.path.join(data_dir, "HR_Analytics.csv")
    file2 = os.path.join(data_dir, "WA_Fn-UseC_-HR-Employee-Attrition.csv")

    dfs = []
    if os.path.exists(file1):
        df1 = pd.read_csv(file1)
        # Drop columns that only exist in HR_Analytics
        for col in DROP_COLUMNS:
            if col in df1.columns:
                df1 = df1.drop(columns=[col])
        dfs.append(df1)

    if os.path.exists(file2):
        df2 = pd.read_csv(file2)
        for col in DROP_COLUMNS:
            if col in df2.columns:
                df2 = df2.drop(columns=[col])
        dfs.append(df2)

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset: handle missing values, duplicates, fix typos."""
    # Fix typo in BusinessTravel
    df["BusinessTravel"] = df["BusinessTravel"].replace("TravelRarely", "Travel_Rarely")

    # Drop duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"[INFO] Dropped {dropped} duplicate rows")

    # Fill missing numerical values with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"[INFO] Cleaned data: {len(df)} rows, {df.isnull().sum().sum()} missing values")
    return df


def encode_features(df: pd.DataFrame, artifacts_dir: str, fit: bool = True):
    """Encode categorical features and the target variable."""
    encoders = {}

    # Encode target: Yes=1, No=0
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    # Label encode categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            if fit:
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            else:
                # Load existing encoder
                le = joblib.load(os.path.join(artifacts_dir, f"encoder_{col}.pkl"))
                df[col] = le.transform(df[col].astype(str))

    if fit:
        os.makedirs(artifacts_dir, exist_ok=True)
        for col, le in encoders.items():
            joblib.dump(le, os.path.join(artifacts_dir, f"encoder_{col}.pkl"))
        print(f"[INFO] Saved {len(encoders)} label encoders")

    return df, encoders


def scale_features(df: pd.DataFrame, artifacts_dir: str, fit: bool = True):
    """Scale numerical features using StandardScaler."""
    feature_cols = [c for c in df.columns if c != TARGET]
    scaler = StandardScaler()

    if fit:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
        joblib.dump(feature_cols, os.path.join(artifacts_dir, "feature_columns.pkl"))
        print(f"[INFO] Scaled {len(feature_cols)} features, saved scaler")
    else:
        scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, scaler, feature_cols


def prepare_data(data_dir: str, artifacts_dir: str, test_size: float = 0.2, random_state: int = 42):
    """Full preprocessing pipeline: load, clean, encode, scale, split."""
    df = load_data(data_dir)
    df = clean_data(df)
    df, encoders = encode_features(df, artifacts_dir, fit=True)
    df, scaler, feature_cols = scale_features(df, artifacts_dir, fit=True)

    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"[INFO] Attrition rate — Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, feature_cols


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml", "artifacts")
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(DATA_DIR, ARTIFACTS_DIR)
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")
