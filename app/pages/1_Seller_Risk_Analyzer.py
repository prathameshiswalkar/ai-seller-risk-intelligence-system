import os
import sys
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Add project root to path in a way that works for Streamlit multipage execution
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from bootstrap import ensure_project_root

PROJECT_ROOT = ensure_project_root()
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# Load environment variables from the project root explicitly
load_dotenv(dotenv_path=ENV_PATH)

import importlib
genai_engine = importlib.import_module('src.inference.genai_engine')
generate_risk_report = genai_engine.generate_risk_report

st.title("Seller Risk Analyzer")

# Check for API key
if not os.getenv("GROQ_API_KEY"):
    st.warning("GROQ_API_KEY environment variable not set. GenAI features disabled.")
    st.info("Please set the environment variable before using this feature.")

revenue = st.number_input("Revenue", 0.0)
late_rate = st.slider("Late Delivery Rate", 0.0, 1.0)
negative_rate = st.slider("Negative Review Rate", 0.0, 1.0)

if st.button("Analyze Seller"):
    
    health_score = 1 - (0.5 * negative_rate + 0.3 * late_rate)
    
    prompt = f"""
    Revenue: {revenue}
    Late Rate: {late_rate}
    Negative Rate: {negative_rate}
    Health Score: {health_score}

    Provide structured risk assessment.
    """

    try:
        report = generate_risk_report(prompt)
        st.subheader("AI Risk Report")
        st.write(report)
    except Exception as e:
        st.error(f"Error generating report: {e}")
        st.info("Make sure GROQ_API_KEY is set correctly.")

if revenue <= 0:
    st.warning("Please enter valid revenue")
