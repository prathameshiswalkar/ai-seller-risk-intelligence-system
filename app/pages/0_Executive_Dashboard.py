import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path - works with Streamlit
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.risk_engine import calculate_risk_level


# Load Seller Data
@st.cache_data
def load_data():
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "seller_master.csv")
    data = pd.read_csv(data_path)
    return data

seller_df = load_data()

st.title("AI Seller Risk Intelligence System")
st.subheader("Seller Selection")

seller_id = st.selectbox("Select a Seller", seller_df["seller_id"].unique())

# Platform Benchmarks
platform_avg_late = seller_df["late_delivery_rate"].mean()
platform_avg_negative = seller_df["negative_rate"].mean()
platform_avg_revenue = seller_df["total_revenue"].mean()

seller = seller_df[seller_df["seller_id"] == seller_id].iloc[0]

st.divider()

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Health Index", round(seller["seller_health_index_v2"], 3))
col2.metric("Negative Rate", f"{round(seller['negative_rate'] * 100, 2)}%")
col3.metric("Late Delivery Rate", f"{round(seller['late_delivery_rate'] * 100, 2)}%")
col4.metric("Revenue", f"${round(seller['total_revenue'], 2)}")

# Benchmark Comparison
st.subheader("Benchmark Comparison")

col5, col6, col7 = st.columns(3)

col5.metric(
    "Late Rate vs Platform",
    f"{round(seller['late_delivery_rate'] * 100, 2)}%",
    delta=f"{round((seller['late_delivery_rate'] - platform_avg_late) * 100, 2)}%"
)

col6.metric(
    "Negative Rate vs Platform",
    round(seller['negative_rate'], 3),
    delta=round(seller['negative_rate'] - platform_avg_negative, 3)
)

col7.metric(
    "Revenue vs Platform",
    round(seller['total_revenue'], 2),
    delta=round(seller['total_revenue'] - platform_avg_revenue, 2)
)

# Visual Comparison
st.subheader("Seller vs Platform Comparison")

comparison_df = pd.DataFrame({
    "Metric": ["Late Delivery Rate", "Negative Rate"],
    "Seller": [seller['late_delivery_rate'], seller['negative_rate']],
    "Platform Average": [platform_avg_late, platform_avg_negative]
})

fig, ax = plt.subplots()
comparison_df.set_index("Metric").plot(kind="bar", ax=ax)
ax.set_ylabel("Rate")
ax.set_title("Seller vs Platform Benchmark")
plt.xticks(rotation=0)

st.pyplot(fig)

# Segment Comparison 
if "seller_segment" in seller_df.columns:
    segment_avg = seller_df[
        seller_df["seller_segment"] == seller["seller_segment"]
    ]["seller_health_index_v2"].mean()

    st.metric(
        "Health Index vs Segment Avg",
        round(seller["seller_health_index_v2"], 3),
        delta=round(seller["seller_health_index_v2"] - segment_avg, 3)
    )

# Risk Classification
risk_level = calculate_risk_level(seller)

if risk_level == "HIGH":
    st.error(f"Risk Level: {risk_level}")
elif risk_level == "MEDIUM":
    st.warning(f"Risk Level: {risk_level}")
else:
    st.success(f"Risk Level: {risk_level}")