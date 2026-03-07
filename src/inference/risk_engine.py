import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.pkl")

xgb = joblib.load(MODEL_PATH)

def predict_late_probability(input_df):
    prob = xgb.predict_proba(input_df)[:, 1]
    return prob[0]

def calculate_risk_level(seller):
    if seller.negative_rate > 0.5:
        return "HIGH"
    if seller.late_delivery_rate > 0.08:
        return "HIGH"
    if seller.seller_health_index_v2 < 0.30:
        return "HIGH"
    if seller.seller_health_index_v2 < 0.50:
        return "MEDIUM"
    return "LOW"