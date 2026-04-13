import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Add project root to path in a way that works for Streamlit multipage execution
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from bootstrap import ensure_project_root

PROJECT_ROOT = ensure_project_root()
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "seller_master.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

st.title("Seller Performance Dashboard")
st.markdown("Comprehensive Seller Health & Risk Intelligence Overview")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 140px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.1rem;
        line-height: 1.1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.subheader("Platform Overview")

total_revenue_m = df["total_revenue"].sum() / 1_000_000
avg_health = df["seller_health_index_v2"].mean()
avg_late_rate = df["late_delivery_rate"].mean() * 100

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Sellers", f"{len(df):,}")
    st.metric("Avg Health Index", f"{avg_health:.3f}")

with col2:
    st.metric("Total Revenue (Millions)", f"${total_revenue_m:,.2f}M")
    st.metric("Avg Late Rate", f"{avg_late_rate:.2f}%")


st.divider()

st.subheader("Risk Segmentation")

df["risk_segment"] = pd.cut(
    df["seller_health_index_v2"],
    bins=[0, 0.3, 0.5, 1],
    labels=["High Risk", "Medium Risk", "Low Risk"]
)

risk_counts = df["risk_segment"].value_counts().sort_index()

col1, col2 = st.columns([1.6, 1])

with col1:
    st.bar_chart(risk_counts)

with col2:
    st.dataframe(risk_counts.rename("Seller Count"))

st.divider()

st.subheader("Seller Detail Analysis")

seller_id = st.selectbox("Select Seller", df["seller_id"].unique())

seller = df[df["seller_id"] == seller_id].iloc[0]

col1, col2 = st.columns(2)

with col1:
    st.metric("Health Index", f"{seller['seller_health_index_v2']:.3f}")
    st.metric("Negative Review Rate", f"{seller['negative_rate'] * 100:.2f}%")

with col2:
    st.metric("Late Delivery Rate", f"{seller['late_delivery_rate'] * 100:.2f}%")
    st.metric("Revenue", f"${seller['total_revenue']:,.2f}")

st.divider()

st.subheader("Seller vs Platform Benchmark")

platform_avg = df.mean(numeric_only=True)

comparison_df = pd.DataFrame({
    "Metric": ["Health Index", "Late Delivery Rate", "Negative Rate"],
    "Seller": [
        seller["seller_health_index_v2"],
        seller["late_delivery_rate"],
        seller["negative_rate"]
    ],
    "Platform Avg": [
        platform_avg["seller_health_index_v2"],
        platform_avg["late_delivery_rate"],
        platform_avg["negative_rate"]
    ]
})

fig, ax = plt.subplots(figsize=(8, 4.8))
comparison_df.set_index("Metric").plot(kind="bar", ax=ax)
ax.set_title("Seller vs Platform")
ax.set_ylabel("Value")
plt.xticks(rotation=0)

st.pyplot(fig)

st.divider()

st.subheader("Revenue vs Health Index")

fig2, ax2 = plt.subplots(figsize=(8, 4.8))
ax2.scatter(df["seller_health_index_v2"], df["total_revenue"])
ax2.set_xlabel("Seller Health Index")
ax2.set_ylabel("Total Revenue")
ax2.set_title("Revenue vs Seller Health")

st.pyplot(fig2)

st.divider()

st.subheader("Top 10 High-Risk Sellers")

top_risk = df.sort_values("seller_health_index_v2").head(10)

st.dataframe(top_risk, use_container_width=True)
