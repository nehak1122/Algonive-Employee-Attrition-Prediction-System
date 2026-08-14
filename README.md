# Employee Attrition Prediction System (EAPS)

EAPS is an ML-powered platform designed to help HR departments predict, explain, and **act on** employee attrition. It doesn't just flag who's at risk — it tests whether a realistic HR intervention (like removing overtime) would actually reduce that risk, closing the loop between prediction and prevention.

## 🚀 Key Features
- **Predictive Modeling**: Trained on real-world HR datasets using Logistic Regression, Random Forest, and XGBoost.
- **Class-Imbalance Handling**: SMOTE oversampling on the training set only, so the model doesn't just learn to predict "stays" for everyone.
- **Explainable AI (SHAP)**: Every prediction comes with a SHAP breakdown showing exactly which features pushed *that specific employee's* risk up or down — not just generic global importance.
- **What-If Intervention Simulator**: The core enhancement — takes an at-risk employee, applies a realistic HR action (remove overtime, raise salary, improve work-life balance, etc.), and re-runs the model to check whether predicted risk *actually drops*. Ranks interventions by real effectiveness instead of guessing.
- **HR Dashboard**: Rich visualizations built with Streamlit and Plotly, including before/after risk comparisons per intervention.
- **FastAPI Backend**: High-performance API for single/batch predictions, SHAP explanations, and intervention simulation.

## 🛠️ Tech Stack
- **Languages**: Python
- **ML/Data**: Scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP, Pandas, NumPy
- **API**: FastAPI, Uvicorn
- **Dashboard**: Streamlit, Plotly
- **Infrastructure**: Git

## 📋 Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the System
Use the provided `run.py` script to train the model and launch both the API and Dashboard:
```bash
python3 run.py
```

- **Dashboard**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📊 Model Performance
The current best model (selected by F1 score, after SMOTE-balanced training) is **Logistic Regression** with:
- **Accuracy**: 75.2%
- **F1 Score**: 44.1%
- **ROC-AUC**: 0.745

All three models (Logistic Regression, Random Forest, XGBoost) are trained and compared every run — see `ml/artifacts/model_metadata.json` for the full comparison table.

## 🧪 What-If Intervention Simulator
Most attrition tools stop at "who is likely to leave and why". EAPS goes one step further:

1. Predict an employee's attrition risk.
2. See a **SHAP explanation** of exactly which of *their* attributes are driving that prediction.
3. Run the **What-If Simulator** to test realistic HR actions (remove overtime, raise salary 15%, improve work-life balance, increase job satisfaction, increase stock options) and see the model's predicted risk **before vs. after** each one — ranked by how much they actually help.

This tests whether *fixing* the problem the model identified genuinely reduces risk, instead of stopping at prediction and explanation.

## 📁 Project Structure
- `ml/data_preprocessing.py`: Load, clean, encode, scale, and SMOTE-balance the data.
- `ml/train_model.py`: Train/compare models, save the best one, compute SHAP + built-in feature importance.
- `ml/shap_explainer.py`: Builds SHAP explainers and formats per-employee / global explanations.
- `ml/intervention_simulator.py`: Applies realistic HR interventions to an employee profile and compares before/after risk.
- `api/`: FastAPI implementation — `/predict`, `/batch-predict`, `/explain`, `/simulate-intervention`, `/feature-importance`, `/shap-importance`.
- `dashboard/`: Streamlit platform, including the Predict, What-If Simulator, and analytics pages.
- `ml/artifacts/`: Saved model, scaler, encoders, and importance data.
