import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved artifacts
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')
numerical_cols = joblib.load('numerical_columns.pkl')

st.title("📊 Customer Churn Prediction App")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure_months = st.number_input("Tenure Months", min_value=0, max_value=72, value=12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=2500.0)

# Derived features
tenure_safe = max(tenure_months, 1)
avg_monthly_spend = total_charges / tenure_safe
has_tech_help = int(online_security == "Yes" or tech_support == "Yes")
has_streaming = int(streaming_tv == "Yes" or streaming_movies == "Yes")
has_bundle = int(phone_service == "Yes" and internet_service != "No" and has_streaming == 1)
is_long_term = int(contract in ["One year", "Two year"])

# Create input dictionary
input_dict = {
    'Gender': gender,
    'Senior Citizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'Tenure Months': tenure_months,
    'Phone Service': phone_service,
    'Multiple Lines': multiple_lines,
    'Internet Service': internet_service,
    'Online Security': online_security,
    'Online Backup': online_backup,
    'Device Protection': device_protection,
    'Tech Support': tech_support,
    'Streaming TV': streaming_tv,
    'Streaming Movies': streaming_movies,
    'Contract': contract,
    'Paperless Billing': paperless_billing,
    'Payment Method': payment_method,
    'Monthly Charges': monthly_charges,
    'Total Charges': total_charges,
    'AvgMonthlySpend': avg_monthly_spend,
    'HasTechHelp': has_tech_help,
    'HasStreaming': has_streaming,
    'HasBundle': has_bundle,
    'IsLongTermContract': is_long_term
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# One-hot encode and align with training columns
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# Scale numerical features using the same columns as training
input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

if st.button("Predict Churn"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    if probability >= 0.4:
        st.success(f"⚠️ This customer is **likely to churn** (Probability: {probability*100:.2f}%)")
    else:
        st.info(f"✅ This customer is **not likely to churn** (Probability: {max(probability*100, 0.0001):.4f}%)")


