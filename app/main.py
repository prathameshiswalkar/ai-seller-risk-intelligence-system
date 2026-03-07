import streamlit as st
import pandas as pd
import sys
import os

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Seller Intelligence Platform",
    layout="wide"
)

# --------------------------------------------------
# Header Section
# --------------------------------------------------

st.title("Seller Intelligence Platform")
st.subheader("AI-Powered Risk & Performance Analytics for E-Commerce Marketplaces")

st.markdown(
"""
Transforming raw marketplace data into actionable business intelligence using  

Machine Learning • Deep Learning • Generative AI • RAG Memory Systems
"""
)

st.divider()

# --------------------------------------------------
# Load Data for Business Metrics
# --------------------------------------------------

DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "seller_master.csv")

if os.path.exists(DATA_PATH):

    df = pd.read_csv(DATA_PATH)

    total_sellers = len(df)
    avg_health = round(df["seller_health_index_v2"].mean(), 3)
    high_risk = len(df[df["seller_health_index_v2"] < 0.4])
    total_revenue = round(df["total_revenue"].sum() / 1_000_000, 2)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sellers", total_sellers)
    col2.metric("Platform Revenue (M)", f"${total_revenue}M")
    col3.metric("Avg Seller Health Index", avg_health)
    col4.metric("High Risk Sellers", high_risk)

else:
    st.warning("Seller data not found.")

st.divider()

# --------------------------------------------------
# Platform Capabilities Section
# --------------------------------------------------

st.subheader("Platform Intelligence Capabilities")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
### Predictive Risk Modeling
- Late Delivery Probability Prediction (XGBoost)
- Seller Health Index Scoring
- Revenue-at-Risk Quantification
- Risk Classification Engine
""")

with col2:
    st.markdown("""
### Sentiment & Behavioral Intelligence
- Multilingual Review Sentiment Analysis (BERT)
- Negative Rate Impact Modeling
- Reputation Risk Scoring
- Customer Experience Signals
""")

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
### Generative AI Advisory
- Structured Risk Reports
- Operational Diagnosis
- Strategic Improvement Roadmap
- Benchmark Comparison
""")

with col4:
    st.markdown("""
### Memory-Enhanced AI (RAG)
- Historical Risk Case Retrieval
- Pattern Matching Across Sellers
- Knowledge-Based Recommendations
- Continuous Learning Architecture
""")

st.divider()

# --------------------------------------------------
# System Architecture
# --------------------------------------------------

st.subheader("System Architecture")

st.markdown("""
Data Layer → Seller Metrics + Reviews  

ML Layer → XGBoost Risk Prediction  

DL Layer → BERT Sentiment Model  

GenAI Layer → Gemini Advisory Engine  

Memory Layer → FAISS Vector Database  

Application Layer → Streamlit Web Interface
""")

st.divider()

# --------------------------------------------------
# Business Impact
# --------------------------------------------------

st.subheader("Business Impact")

st.markdown("""
- Proactive Identification of High-Risk Sellers  
- Revenue Protection Through Early Intervention  
- Customer Satisfaction Intelligence  
- Operational Risk Mitigation  
- AI-Augmented Decision Support System  
""")

st.divider()

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("""
---

Developed as an End-to-End AI Business Intelligence System  

Capstone Project — ML + DL + GenAI + Web Application
""")