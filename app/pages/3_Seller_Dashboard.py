import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path - works with Streamlit
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


st.title("Seller Performance Dashboard")
st.markdown("Comprehensive Seller Health & Risk Intelligence Overview")


# Load Data (Safe Path)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "seller_master.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()


# Platform KPIs

st.subheader("Platform Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sellers", len(df))
col2.metric("Total Revenue", f"${df['total_revenue'].sum():,.2f}")
col3.metric("Avg Health Index", round(df["seller_health_index_v2"].mean(), 3))
col4.metric("Avg Late Rate", f"{df['late_delivery_rate'].mean()*100:.2f}%")

st.divider()


# Risk Segmentation

st.subheader("Risk Segmentation")

df["risk_segment"] = pd.cut(
    df["seller_health_index_v2"],
    bins=[0, 0.3, 0.5, 1],
    labels=["High Risk", "Medium Risk", "Low Risk"]
)

risk_counts = df["risk_segment"].value_counts().sort_index()

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(risk_counts)

with col2:
    st.dataframe(risk_counts.rename("Seller Count"))

st.divider()


# Seller Selector

st.subheader("Seller Detail Analysis")

seller_id = st.selectbox("Select Seller", df["seller_id"].unique())

seller = df[df["seller_id"] == seller_id].iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Health Index", round(seller["seller_health_index_v2"], 3))
col2.metric("Late Delivery Rate", f"{seller['late_delivery_rate']*100:.2f}%")
col3.metric("Negative Review Rate", f"{seller['negative_rate']*100:.2f}%")
col4.metric("Revenue", f"${seller['total_revenue']:,.2f}")

st.divider()


# Benchmark Comparison

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

fig, ax = plt.subplots()
comparison_df.set_index("Metric").plot(kind="bar", ax=ax)
ax.set_title("Seller vs Platform")
ax.set_ylabel("Value")
plt.xticks(rotation=0)

st.pyplot(fig)

st.divider()


# Revenue vs Health Scatter Plot

st.subheader("Revenue vs Health Index")

fig2, ax2 = plt.subplots()
ax2.scatter(df["seller_health_index_v2"], df["total_revenue"])
ax2.set_xlabel("Seller Health Index")
ax2.set_ylabel("Total Revenue")
ax2.set_title("Revenue vs Seller Health")

st.pyplot(fig2)

st.divider()


# Top Risk Sellers

st.subheader("Top 10 High-Risk Sellers")

top_risk = df.sort_values("seller_health_index_v2").head(10)

st.dataframe(top_risk)