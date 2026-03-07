import sys
import os
import streamlit as st
import pandas as pd

# Add project root to path - works with Streamlit
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.sentiment_engine import analyze_sentiment



st.title("Review Sentiment Intelligence")
st.markdown("Deep Learning Powered Seller Review Analysis (BERT-based Model)")

review = st.text_area("Enter Seller Review", height=150)

if st.button("Analyze Sentiment"):

    if not review.strip():
        st.warning("Please enter a review")
    else:
        try:
            result = analyze_sentiment(review)

            sentiment = result["sentiment"]
            rating = result["rating"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]

            st.divider()

            col1, col2, col3 = st.columns(3)

            # Sentiment Display
            if sentiment == "positive":
                col1.success("Sentiment: POSITIVE")
                risk_impact = "Low Reputation Risk"
            elif sentiment == "negative":
                col1.error("Sentiment: NEGATIVE")
                risk_impact = "High Reputation Risk"
            else:
                col1.warning("Sentiment: NEUTRAL")
                risk_impact = "Moderate Reputation Risk"

            col2.metric("Star Rating (1–5)", rating)
            col3.metric("Model Confidence", f"{confidence*100:.2f}%")

            st.divider()

            # Probability Visualization
            st.subheader("Model Probability Distribution")

            prob_df = pd.DataFrame({
                "Star Rating": [1, 2, 3, 4, 5],
                "Probability": probabilities
            })

            st.bar_chart(prob_df.set_index("Star Rating"))

            st.divider()

            # Business Interpretation
            st.subheader("Business Interpretation")

            if sentiment == "positive":
                st.markdown("""
                - Strong customer satisfaction signal  
                - Supports seller health stability  
                - Low churn and revenue risk  
                - Positive brand reinforcement  
                """)
            elif sentiment == "negative":
                st.markdown("""
                - High dissatisfaction signal  
                - Potential increase in negative review rate  
                - Risk of revenue loss if pattern continues  
                - Requires operational investigation  
                """)
            else:
                st.markdown("""
                - Moderate customer experience  
                - Monitor review trend over time  
                - Neutral short-term risk impact  
                """)

            st.metric("Reputation Risk Impact", risk_impact)

        except Exception as e:
            st.error(f"Model Error: {e}")