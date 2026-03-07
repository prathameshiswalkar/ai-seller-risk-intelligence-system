import sys
import os
import streamlit as st

# Adding project root to path - works with Streamlit
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.genai_engine import vector_store

st.title("Historical Risk Memory")

query = st.text_input("Search Similar Risk Cases")

if st.button("Search"):
    if vector_store is None:
        st.error("Vector store not available. Please check configuration.")
    elif not query.strip():
        st.warning("Please enter search query")
    else:
        try:
            docs = vector_store.similarity_search(query, k=3)
            for doc in docs:
                st.write(doc.page_content)
        except Exception as e:
            st.error(f"Search error: {e}")