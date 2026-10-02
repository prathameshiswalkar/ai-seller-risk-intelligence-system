import joblib
import os
import numpy as np
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


# --------------------------------------------------
# Predict Late Delivery Probability
# --------------------------------------------------

def predict_late_probability(input_df: pd.DataFrame):

    # These are the order-level fields used to train xgb_model.pkl in
    # notebook/Late_delivery_analysis.ipynb. Keep the inference contract in
    # sync with that model rather than the separate seller summary dataset.
    required_features = [
        "estimated_delivery_days",
        "order_month",
        "order_weekday",
        "total_payment_value",
        "avg_installments",
        "total_price",
        "total_freight",
        "total_items",
        "seller_late_rate",
    ]

    for col in required_features:
        if col not in input_df.columns:
            raise ValueError(f"Missing feature column: {col}")

    if input_df.empty:
        raise ValueError("At least one seller row is required for prediction")

    # Load only when prediction is requested. Dashboard pages that only need
    # rule-based risk classification should remain usable without the artifact.
    model = load_xgb_model()
    probabilities = np.asarray(model.predict_proba(input_df))
    if probabilities.ndim != 2 or probabilities.shape[0] != len(input_df):
        raise ValueError("Model returned an invalid probability array")
    if probabilities.shape[1] < 2:
        raise ValueError("Model must return probabilities for both classes")

    return float(probabilities[0, 1])


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
