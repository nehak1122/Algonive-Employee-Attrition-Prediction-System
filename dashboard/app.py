"""
EAPS - Employee Attrition Prediction System Dashboard
Rich Streamlit app with multiple pages for HR analytics.
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Config ---
st.set_page_config(
    page_title="EAPS - Employee Attrition Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml", "artifacts")

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .stMetric label {
        color: #a0aec0 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 50%, #24243e 100%);
    }

    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
    }

    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .hero-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .hero-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .risk-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(255, 65, 108, 0.4);
    }

    .risk-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.4);
    }

    .risk-low {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: #1a1a2e;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(67, 233, 123, 0.4);
    }

    .info-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading ---
@st.cache_data
def load_raw_data():
    """Load raw data for exploration."""
    dfs = []
    f1 = os.path.join(BASE_DIR, "HR_Analytics.csv")
    f2 = os.path.join(BASE_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")

    if os.path.exists(f1):
        df1 = pd.read_csv(f1)
        dfs.append(df1)
    if os.path.exists(f2):
        df2 = pd.read_csv(f2)
        dfs.append(df2)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df["BusinessTravel"] = df["BusinessTravel"].replace("TravelRarely", "Travel_Rarely")
        df = df.drop_duplicates()
        return df
    return pd.DataFrame()


@st.cache_data
def load_model_metadata():
    """Load model metadata from artifacts."""
    meta_path = os.path.join(ARTIFACTS_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_feature_importance():
    """Load feature importance from artifacts."""
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.json")
    if os.path.exists(fi_path):
        with open(fi_path, "r") as f:
            return json.load(f)
    return None


# --- Color Palette ---
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "accent": "#43e97b",
    "danger": "#ff416c",
    "warning": "#f093fb",
    "bg_dark": "#1a1a2e",
    "text": "#e2e8f0",
}

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_SEQUENCE = ["#667eea", "#764ba2", "#43e97b", "#ff416c", "#f093fb", "#38f9d7", "#ffd700", "#ff6b6b"]


# --- Sidebar ---
st.sidebar.markdown("## 👥 EAPS")
st.sidebar.markdown("**Employee Attrition Prediction System**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Department Analysis", "🔍 Feature Insights",
     "🎯 Predict", "📂 Batch Predict", "📋 Data Explorer"],
    label_visibility="collapsed",
)

# Model Info in sidebar
meta = load_model_metadata()
if meta:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Info")
    st.sidebar.markdown(f"**Model:** {meta['best_model']}")
    st.sidebar.markdown(f"**Accuracy:** {meta['best_metrics']['accuracy']:.1%}")
    st.sidebar.markdown(f"**F1 Score:** {meta['best_metrics']['f1_score']:.1%}")
    st.sidebar.markdown(f"**ROC-AUC:** {meta['best_metrics']['roc_auc']:.1%}")


# ==================== PAGES ====================

if page == "🏠 Overview":
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1>👥 Employee Attrition Prediction</h1>
        <p>AI-powered insights to help HR teams retain top talent</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_raw_data()
    if df.empty:
        st.error("No data loaded. Ensure CSV files are in the EAPS directory.")
        st.stop()

    # KPI Cards
    total_emp = len(df)
    attrition_yes = len(df[df["Attrition"] == "Yes"])
    attrition_rate = attrition_yes / total_emp * 100
    avg_age = df["Age"].mean()
    avg_income = df["MonthlyIncome"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", f"{total_emp:,}")
    col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta=f"{attrition_yes} employees")
    col3.metric("Avg. Age", f"{avg_age:.0f} yrs")
    col4.metric("Avg. Monthly Income", f"₹{avg_income:,.0f}")

    st.markdown("---")

    # Charts row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Attrition Donut
        attrition_counts = df["Attrition"].value_counts()
        fig_donut = go.Figure(data=[go.Pie(
            labels=attrition_counts.index,
            values=attrition_counts.values,
            hole=0.65,
            marker=dict(colors=["#43e97b", "#ff416c"]),
            textinfo="label+percent",
            textfont=dict(size=14, color="white"),
        )])
        fig_donut.update_layout(
            title="Attrition Distribution",
            template=PLOTLY_TEMPLATE,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            showlegend=True,
            legend=dict(font=dict(size=12)),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart_col2:
        # Age Distribution by Attrition
        fig_age = px.histogram(
            df, x="Age", color="Attrition",
            barmode="overlay", nbins=30,
            color_discrete_map={"No": "#667eea", "Yes": "#ff416c"},
            opacity=0.7,
        )
        fig_age.update_layout(
            title="Age Distribution by Attrition",
            template=PLOTLY_TEMPLATE,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            xaxis_title="Age",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # Second row
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        # Monthly Income by Attrition - violin
        fig_income = px.violin(
            df, x="Attrition", y="MonthlyIncome",
            color="Attrition",
            color_discrete_map={"No": "#667eea", "Yes": "#ff416c"},
            box=True,
        )
        fig_income.update_layout(
            title="Monthly Income Distribution",
            template=PLOTLY_TEMPLATE,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
        )
        st.plotly_chart(fig_income, use_container_width=True)

    with chart_col4:
        # Overtime Impact
        ot_data = df.groupby(["OverTime", "Attrition"]).size().reset_index(name="Count")
        fig_ot = px.bar(
            ot_data, x="OverTime", y="Count", color="Attrition",
            barmode="group",
            color_discrete_map={"No": "#667eea", "Yes": "#ff416c"},
        )
        fig_ot.update_layout(
            title="Overtime vs Attrition",
            template=PLOTLY_TEMPLATE,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
        )
        st.plotly_chart(fig_ot, use_container_width=True)


elif page == "📊 Department Analysis":
    st.markdown("## 📊 Department-wise Attrition Analysis")
    st.markdown("---")

    df = load_raw_data()
    if df.empty:
        st.stop()

    # Department attrition
    dept_data = df.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
    dept_total = df.groupby("Department").size().reset_index(name="Total")
    dept_yes = df[df["Attrition"] == "Yes"].groupby("Department").size().reset_index(name="Left")
    dept_rate = dept_total.merge(dept_yes, on="Department", how="left")
    dept_rate["Left"] = dept_rate["Left"].fillna(0)
    dept_rate["Attrition Rate (%)"] = (dept_rate["Left"] / dept_rate["Total"] * 100).round(1)

    col1, col2 = st.columns(2)

    with col1:
        fig_dept = px.bar(
            dept_data, x="Department", y="Count", color="Attrition",
            barmode="group",
            color_discrete_map={"No": "#667eea", "Yes": "#ff416c"},
        )
        fig_dept.update_layout(
            title="Attrition by Department",
            template=PLOTLY_TEMPLATE,
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
        )
        st.plotly_chart(fig_dept, use_container_width=True)

    with col2:
        fig_rate = px.bar(
            dept_rate, x="Department", y="Attrition Rate (%)",
            color="Attrition Rate (%)",
            color_continuous_scale=["#43e97b", "#ffd700", "#ff416c"],
        )
        fig_rate.update_layout(
            title="Attrition Rate by Department",
            template=PLOTLY_TEMPLATE,
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
        )
        st.plotly_chart(fig_rate, use_container_width=True)

    # Job Role analysis
    st.markdown("### 👔 Attrition by Job Role")
    role_data = df.groupby(["JobRole", "Attrition"]).size().reset_index(name="Count")
    role_total = df.groupby("JobRole").size().reset_index(name="Total")
    role_yes = df[df["Attrition"] == "Yes"].groupby("JobRole").size().reset_index(name="Left")
    role_rate = role_total.merge(role_yes, on="JobRole", how="left")
    role_rate["Left"] = role_rate["Left"].fillna(0)
    role_rate["Attrition Rate (%)"] = (role_rate["Left"] / role_rate["Total"] * 100).round(1)
    role_rate = role_rate.sort_values("Attrition Rate (%)", ascending=True)

    fig_role = px.bar(
        role_rate, y="JobRole", x="Attrition Rate (%)",
        orientation="h",
        color="Attrition Rate (%)",
        color_continuous_scale=["#43e97b", "#ffd700", "#ff416c"],
    )
    fig_role.update_layout(
        template=PLOTLY_TEMPLATE,
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
    )
    st.plotly_chart(fig_role, use_container_width=True)

    # Marital Status & Business Travel
    col3, col4 = st.columns(2)

    with col3:
        ms_data = df.groupby(["MaritalStatus", "Attrition"]).size().reset_index(name="Count")
        fig_ms = px.bar(
            ms_data, x="MaritalStatus", y="Count", color="Attrition",
            barmode="group",
            color_discrete_map={"No": "#667eea", "Yes": "#ff416c"},
        )
        fig_ms.update_layout(
            title="Marital Status vs Attrition",
            template=PLOTLY_TEMPLATE,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
        )
        st.plotly_chart(fig_ms, use_container_width=True)

    with col4:
        bt_data = df.groupby(["BusinessTravel", "Attrition"]).size().reset_index(name="Count")
        fig_bt = px.bar(
            bt_data, x="BusinessTravel", y="Count", color="Attrition",
            barmode="group",
            color_discrete_map={"No": "#667eea", "Yes": "#ff416c"},
        )
        fig_bt.update_layout(
            title="Business Travel vs Attrition",
            template=PLOTLY_TEMPLATE,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
        )
        st.plotly_chart(fig_bt, use_container_width=True)


elif page == "🔍 Feature Insights":
    st.markdown("## 🔍 Feature Insights")
    st.markdown("---")

    fi = load_feature_importance()
    df = load_raw_data()

    if fi:
        # Feature importance bar chart
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
        fi_df = fi_df.sort_values("Importance", ascending=True).tail(15)

        fig_fi = px.bar(
            fi_df, y="Feature", x="Importance",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#667eea", "#764ba2", "#ff416c"],
        )
        fig_fi.update_layout(
            title="Top 15 Feature Importances",
            template=PLOTLY_TEMPLATE,
            height=550,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.warning("Feature importance not available. Train the model first.")

    if not df.empty:
        st.markdown("### 📈 Correlation Heatmap")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Select important numeric columns
        important_cols = [
            "Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
            "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
            "DistanceFromHome", "PercentSalaryHike", "JobSatisfaction",
            "EnvironmentSatisfaction", "WorkLifeBalance", "JobLevel",
        ]
        available_cols = [c for c in important_cols if c in num_cols]
        corr = df[available_cols].corr()

        fig_corr = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            text_auto=".2f",
        )
        fig_corr.update_layout(
            title="Feature Correlation Heatmap",
            template=PLOTLY_TEMPLATE,
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"], size=10),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Model comparison
    if meta and "all_results" in meta:
        st.markdown("### 🏆 Model Comparison")
        results_df = pd.DataFrame(meta["all_results"]).T
        results_df.index.name = "Model"
        results_df = results_df.reset_index()

        fig_comp = go.Figure()
        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        colors = ["#667eea", "#764ba2", "#43e97b", "#ff416c", "#ffd700"]

        for i, metric in enumerate(metrics_to_plot):
            fig_comp.add_trace(go.Bar(
                name=metric.replace("_", " ").title(),
                x=results_df["Model"],
                y=results_df[metric],
                marker_color=colors[i],
            ))

        fig_comp.update_layout(
            barmode="group",
            title="Model Performance Comparison",
            template=PLOTLY_TEMPLATE,
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            yaxis_title="Score",
        )
        st.plotly_chart(fig_comp, use_container_width=True)


elif page == "🎯 Predict":
    st.markdown("## 🎯 Predict Employee Attrition")
    st.markdown("Enter employee details to get an attrition risk prediction.")
    st.markdown("---")

    df = load_raw_data()

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Personal Info")
            age = st.slider("Age", 18, 65, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            distance = st.slider("Distance From Home (km)", 1, 30, 10)
            education = st.slider("Education Level", 1, 5, 3, help="1=Below College, 5=Doctor")
            edu_field = st.selectbox("Education Field", [
                "Life Sciences", "Medical", "Marketing",
                "Technical Degree", "Human Resources", "Other"
            ])

        with col2:
            st.markdown("#### 💼 Job Info")
            department = st.selectbox("Department", [
                "Research & Development", "Sales", "Human Resources"
            ])
            job_role = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative",
                "Manager", "Sales Representative", "Research Director",
                "Human Resources"
            ])
            job_level = st.slider("Job Level", 1, 5, 2)
            job_involvement = st.slider("Job Involvement", 1, 4, 3)
            job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
            business_travel = st.selectbox("Business Travel", [
                "Travel_Rarely", "Travel_Frequently", "Non-Travel"
            ])

        with col3:
            st.markdown("#### 💰 Compensation & Experience")
            monthly_income = st.number_input("Monthly Income (₹)", 1000, 200000, 5000, step=500)
            overtime = st.selectbox("Overtime", ["No", "Yes"])
            total_years = st.slider("Total Working Years", 0, 40, 10)
            years_company = st.slider("Years at Company", 0, 40, 5)
            years_role = st.slider("Years in Current Role", 0, 20, 3)
            years_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
            years_mgr = st.slider("Years With Current Manager", 0, 20, 3)
            env_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
            work_life = st.slider("Work-Life Balance", 1, 4, 3)
            stock_option = st.slider("Stock Option Level", 0, 3, 1)

        submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

    if submitted:
        payload = {
            "Age": age,
            "BusinessTravel": business_travel,
            "DailyRate": 800,
            "Department": department,
            "DistanceFromHome": distance,
            "Education": education,
            "EducationField": edu_field,
            "EnvironmentSatisfaction": env_satisfaction,
            "Gender": gender,
            "HourlyRate": 65,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobRole": job_role,
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": 15000,
            "NumCompaniesWorked": 2,
            "OverTime": overtime,
            "PercentSalaryHike": 14,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": stock_option,
            "TotalWorkingYears": total_years,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": work_life,
            "YearsAtCompany": years_company,
            "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": years_promo,
            "YearsWithCurrManager": years_mgr,
        }

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                st.markdown("---")
                st.markdown("### 📊 Prediction Result")

                r_col1, r_col2, r_col3 = st.columns(3)

                with r_col1:
                    risk_class = result["risk_level"].lower()
                    st.markdown(f"""
                    <div class="risk-{risk_class}">
                        <div style="font-size:0.9rem; opacity:0.8;">Risk Level</div>
                        <div style="font-size:2rem; font-weight:700;">{result['risk_level']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with r_col2:
                    st.metric("Attrition Prediction", result["attrition_risk"])

                with r_col3:
                    st.metric("Probability", f"{result['probability']:.1%}")

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["probability"] * 100,
                    title={"text": "Attrition Risk Score", "font": {"color": "white"}},
                    number={"suffix": "%", "font": {"color": "white"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "white"},
                        "bar": {"color": "#667eea"},
                        "steps": [
                            {"range": [0, 40], "color": "#43e97b"},
                            {"range": [40, 70], "color": "#ffd700"},
                            {"range": [70, 100], "color": "#ff416c"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 4},
                            "thickness": 0.8,
                            "value": result["probability"] * 100,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            else:
                st.error(f"API Error: {resp.status_code} — {resp.text}")
        except requests.ConnectionError:
            st.error("⚠️ Cannot connect to the API. Make sure the FastAPI server is running on port 8000.")
            st.code("cd EAPS && uvicorn api.main:app --port 8000 --reload", language="bash")


elif page == "📂 Batch Predict":
    st.markdown("## 📂 Batch Prediction")
    st.markdown("Upload a CSV file to predict attrition for multiple employees.")
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_preview = pd.read_csv(uploaded)
        st.markdown(f"**{len(df_preview)} employees found in file**")
        st.dataframe(df_preview.head(10), use_container_width=True)

        if st.button("🔮 Run Batch Prediction", use_container_width=True):
            uploaded.seek(0)
            try:
                resp = requests.post(
                    f"{API_URL}/batch-predict",
                    files={"file": ("employees.csv", uploaded.getvalue(), "text/csv")},
                    timeout=60,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    pred_df = pd.DataFrame(result["predictions"])

                    st.markdown("### 📊 Results")
                    col1, col2, col3 = st.columns(3)
                    total = len(pred_df)
                    high_risk = len(pred_df[pred_df["risk_level"] == "High"])
                    med_risk = len(pred_df[pred_df["risk_level"] == "Medium"])

                    col1.metric("Total Employees", total)
                    col2.metric("🔴 High Risk", high_risk)
                    col3.metric("🟡 Medium Risk", med_risk)

                    # Risk distribution
                    risk_counts = pred_df["risk_level"].value_counts()
                    fig_risk = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        color=risk_counts.index,
                        color_discrete_map={"High": "#ff416c", "Medium": "#ffd700", "Low": "#43e97b"},
                        hole=0.5,
                    )
                    fig_risk.update_layout(
                        title="Risk Distribution",
                        template=PLOTLY_TEMPLATE,
                        height=400,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=COLORS["text"]),
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)

                    # Results table
                    st.dataframe(
                        pred_df.style.apply(
                            lambda x: [
                                "background-color: #ff416c33" if v == "High"
                                else "background-color: #ffd70033" if v == "Medium"
                                else "background-color: #43e97b33" if v == "Low"
                                else "" for v in x
                            ], subset=["risk_level"]
                        ),
                        use_container_width=True,
                    )
                else:
                    st.error(f"API Error: {resp.status_code}")
            except requests.ConnectionError:
                st.error("⚠️ Cannot connect to the API server.")


elif page == "📋 Data Explorer":
    st.markdown("## 📋 Data Explorer")
    st.markdown("---")

    df = load_raw_data()
    if df.empty:
        st.stop()

    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        dept_filter = st.multiselect("Department", df["Department"].unique().tolist(), default=df["Department"].unique().tolist())
    with filter_col2:
        attrition_filter = st.multiselect("Attrition", ["Yes", "No"], default=["Yes", "No"])
    with filter_col3:
        age_range = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (18, 60))

    filtered = df[
        (df["Department"].isin(dept_filter)) &
        (df["Attrition"].isin(attrition_filter)) &
        (df["Age"] >= age_range[0]) &
        (df["Age"] <= age_range[1])
    ]

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} employees**")
    st.dataframe(filtered, use_container_width=True, height=600)

    # Download
    csv = filtered.to_csv(index=False)
    st.download_button("📥 Download Filtered Data", csv, "filtered_employees.csv", "text/csv")
