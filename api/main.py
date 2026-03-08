"""
FastAPI Backend for EAPS
Serves model predictions and metadata.
"""

import os
import sys
import json
import io
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml", "artifacts")

app = FastAPI(
    title="EAPS - Employee Attrition Prediction System",
    description="Predict employee attrition risk using ML models",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model artifacts at startup ---
model = None
scaler = None
feature_columns = None
encoders = {}
metadata = None
feature_importance = None


def load_artifacts():
    """Load all saved model artifacts."""
    global model, scaler, feature_columns, encoders, metadata, feature_importance

    model = joblib.load(os.path.join(ARTIFACTS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))

    with open(os.path.join(ARTIFACTS_DIR, "model_metadata.json"), "r") as f:
        metadata = json.load(f)

    with open(os.path.join(ARTIFACTS_DIR, "feature_importance.json"), "r") as f:
        feature_importance = json.load(f)

    # Load label encoders
    categorical_cols = [
        "BusinessTravel", "Department", "EducationField",
        "Gender", "JobRole", "MaritalStatus", "OverTime"
    ]
    for col in categorical_cols:
        enc_path = os.path.join(ARTIFACTS_DIR, f"encoder_{col}.pkl")
        if os.path.exists(enc_path):
            encoders[col] = joblib.load(enc_path)

    print(f"[INFO] Loaded model: {metadata['best_model']}")
    print(f"[INFO] Features: {len(feature_columns)}")


@app.on_event("startup")
async def startup_event():
    try:
        load_artifacts()
    except Exception as e:
        print(f"[WARNING] Could not load model artifacts: {e}")
        print("[WARNING] Train the model first: python ml/train_model.py")


# --- Pydantic Models ---
class EmployeeInput(BaseModel):
    Age: int = 35
    BusinessTravel: str = "Travel_Rarely"
    DailyRate: int = 800
    Department: str = "Research & Development"
    DistanceFromHome: int = 10
    Education: int = 3
    EducationField: str = "Life Sciences"
    EnvironmentSatisfaction: int = 3
    Gender: str = "Male"
    HourlyRate: int = 65
    JobInvolvement: int = 3
    JobLevel: int = 2
    JobRole: str = "Research Scientist"
    JobSatisfaction: int = 3
    MaritalStatus: str = "Married"
    MonthlyIncome: int = 5000
    MonthlyRate: int = 15000
    NumCompaniesWorked: int = 2
    OverTime: str = "No"
    PercentSalaryHike: int = 14
    PerformanceRating: int = 3
    RelationshipSatisfaction: int = 3
    StockOptionLevel: int = 1
    TotalWorkingYears: int = 10
    TrainingTimesLastYear: int = 3
    WorkLifeBalance: int = 3
    YearsAtCompany: int = 5
    YearsInCurrentRole: int = 3
    YearsSinceLastPromotion: int = 1
    YearsWithCurrManager: int = 3


class PredictionResponse(BaseModel):
    attrition_risk: str
    probability: float
    risk_level: str
    top_factors: dict


def preprocess_input(data: dict) -> pd.DataFrame:
    """Preprocess a single input dict into model-ready format."""
    df = pd.DataFrame([data])

    # Fix typos
    if "BusinessTravel" in df.columns:
        df["BusinessTravel"] = df["BusinessTravel"].replace("TravelRarely", "Travel_Rarely")

    # Encode categorical columns
    for col, enc in encoders.items():
        if col in df.columns:
            try:
                df[col] = enc.transform(df[col].astype(str))
            except ValueError:
                # Handle unseen labels
                df[col] = 0

    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only the feature columns in order
    df = df[feature_columns]

    # Scale
    df[feature_columns] = scaler.transform(df[feature_columns])

    return df


# --- Endpoints ---

@app.get("/")
async def root():
    return {
        "message": "EAPS - Employee Attrition Prediction System",
        "status": "running",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(employee: EmployeeInput):
    """Predict attrition for a single employee."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    data = employee.model_dump()
    df = preprocess_input(data)

    probability = float(model.predict_proba(df)[0][1])
    prediction = "Yes" if probability >= 0.5 else "No"

    if probability >= 0.7:
        risk_level = "High"
    elif probability >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Top 5 factors
    top_5 = dict(list(feature_importance.items())[:5])

    return PredictionResponse(
        attrition_risk=prediction,
        probability=round(probability, 4),
        risk_level=risk_level,
        top_factors=top_5,
    )


@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """Batch predict from uploaded CSV."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    content = await file.read()
    df_raw = pd.read_csv(io.BytesIO(content))

    results = []
    for idx, row in df_raw.iterrows():
        data = row.to_dict()
        try:
            df_processed = preprocess_input(data)
            probability = float(model.predict_proba(df_processed)[0][1])
            prediction = "Yes" if probability >= 0.5 else "No"
            if probability >= 0.7:
                risk_level = "High"
            elif probability >= 0.4:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            results.append({
                "index": idx,
                "attrition_risk": prediction,
                "probability": round(probability, 4),
                "risk_level": risk_level,
            })
        except Exception as e:
            results.append({
                "index": idx,
                "attrition_risk": "Error",
                "probability": 0,
                "risk_level": "Unknown",
                "error": str(e),
            })

    return {"total": len(results), "predictions": results}


@app.get("/model-info")
async def model_info():
    """Return model metadata."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")
    return metadata


@app.get("/feature-importance")
async def get_feature_importance():
    """Return feature importance values."""
    if feature_importance is None:
        raise HTTPException(status_code=503, detail="Feature importance not loaded.")
    return feature_importance
