# Employee Attrition Prediction System (EAPS)

EAPS is an ML-powered platform designed to help HR departments predict and understand employee attrition. It uses classification models to identify "at-risk" employees and provides actionable insights through an interactive dashboard.

## 🚀 Key Features
- **Predictive Modeling**: Trained on real-world HR datasets using Logistic Regression, Random Forest, and XGBoost.
- **HR Dashboard**: Rich visualizations built with Streamlit and Plotly.
- **FastAPI Backend**: High-performance API for single and batch predictions.
- **Insightful Analytics**: Feature importance analysis to see what drives attrition (e.g., Overtime, Monthly Income).

## 🛠️ Tech Stack
- **Languages**: Python
- **ML/Data**: Scikit-learn, XGBoost, Pandas, NumPy
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
The current best model is **Logistic Regression** with:
- **Accuracy**: 73.5%
- **F1 Score**: 43.4%
- **ROC-AUC**: 0.748

## 📁 Project Structure
- `ml/`: Data preprocessing and training scripts.
- `api/`: FastAPI implementation.
- `dashboard/`: Streamlit platform.
- `ml/artifacts/`: Saved model, scaler, and encoders.
