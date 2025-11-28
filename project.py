pip3 install xgboost
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import plotly.express as px
from sklearn.model_selection import train_test_split
import shap

# =============================
# 1. Load and Preprocess Data
# =============================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

# Replace service categories
df.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)

# Encode Churn
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop ID
df = df.drop("customerID", axis=1)

# Encode features
X = pd.get_dummies(df.drop("Churn", axis=1), drop_first=True)
y = df["Churn"]

# Train model
model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
)
model.fit(X, y)

FEATURE_NAMES = X.columns


# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")
st.title("📊 Telco Customer Churn Predictor (Streamlit Version)")
st.write("This app predicts if a customer is likely to churn using XGBoost.")

st.sidebar.header("Customer Input Parameters")

# Sidebar Inputs
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges ($)", 18, 120, 70)

contract = st.sidebar.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)

internet = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

# Build input dictionary
input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly],
    'TotalCharges': [monthly * tenure],
    'Contract': [contract],
    'InternetService': [internet],
    'PaymentMethod': [payment],
    'gender': ['Male'],
    'SeniorCitizen': [0],
    'Partner': ['No'],
    'Dependents': ['No'],
    'PhoneService': ['Yes'],
    'MultipleLines': ['No'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['No'],
    'StreamingMovies': ['No'],
    'PaperlessBilling': ['Yes']
})

# Encode exactly like training
input_encoded = pd.get_dummies(input_data, drop_first=True)
input_encoded = input_encoded.reindex(columns=FEATURE_NAMES, fill_value=0)

# Predict
prob = model.predict_proba(input_encoded)[0, 1]
risk = "🔥 High Risk - Likely to Churn" if prob > 0.5 else "🟢 Low Risk - Likely to Stay"
color = "red" if prob > 0.5 else "green"

st.subheader("Prediction Result")
st.markdown(f"### **{risk} ({prob:.1%} chance of leaving)**")

# Pie chart
fig = px.pie(
    values=[prob, 1 - prob],
    names=["Churn Risk", "Will Stay"],
    hole=0.6,
)
fig.update_traces(marker=dict(colors=[color, "lightgray"]))
fig.update_layout(title="Churn Prediction Probability")

st.plotly_chart(fig, use_container_width=True)

st.write("---")

# =============================
# SHAP EXPLAINABILITY
# =============================
st.subheader("🔍 SHAP Feature Importance")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.write("Top factors contributing to churn:")

# SHAP summary plot
st.set_option('deprecation.showPyplotGlobalUse', False)
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot()

st.write("---")
st.write("Built with Streamlit + XGBoost + SHAP")


