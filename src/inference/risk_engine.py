import joblib
import os
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.pkl")


# --------------------------------------------------
# Load XGBoost Model
# --------------------------------------------------

@st.cache_resource
def load_xgb_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    return model


xgb_model = load_xgb_model()


# --------------------------------------------------
# Predict Late Delivery Probability
# --------------------------------------------------

def predict_late_probability(input_df: pd.DataFrame):

    required_features = [
        "total_revenue",
        "late_delivery_rate",
        "negative_rate",
        "seller_health_index_v2"
    ]

    for col in required_features:
        if col not in input_df.columns:
            raise ValueError(f"Missing feature column: {col}")

    prob = xgb_model.predict_proba(input_df)[:, 1]

    return float(prob[0])


# --------------------------------------------------
# Risk Level Calculation
# --------------------------------------------------

def calculate_risk_level(seller: dict):

    negative = seller.get("negative_rate", 0)
    late = seller.get("late_delivery_rate", 0)
    health = seller.get("seller_health_index_v2", 1)

    if negative > 0.5:
        return "HIGH"

    if late > 0.08:
        return "HIGH"

    if health < 0.30:
        return "HIGH"

    if health < 0.50:
        return "MEDIUM"

    return "LOW"