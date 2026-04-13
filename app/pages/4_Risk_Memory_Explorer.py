import os
import sys
import streamlit as st

# Add project root to path in a way that works for Streamlit multipage execution
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from bootstrap import ensure_project_root

PROJECT_ROOT = ensure_project_root()

import importlib

vector_store = None
import_error = None

try:
    genai_engine = importlib.import_module('src.inference.genai_engine')
    vector_store = genai_engine.get_vector_store()
    INDEX_PATH = genai_engine.INDEX_PATH
except Exception as e:
    vector_store = None
    import_error = str(e)

st.title("Historical Risk Memory")
st.caption("Search similar historical seller risk cases using vector memory")

query = st.text_input("Search Similar Risk Cases")

with st.expander("System Debug Info"):
    st.write("Project Root:", PROJECT_ROOT)
    st.write("Vector store loaded:", vector_store is not None)
    if import_error:
        st.write("Import error:", import_error)

if st.button("Search"):
    if vector_store is None:
        st.error(
            "Vector store not available. Either the FAISS index hasn’t been built "
            "or it’s not checked in; see log messages above."
        )
        if import_error:
            st.write("Import error:", import_error)
    elif not query.strip():
        st.warning("Please enter a search query.")
    else:
        try:
            results = vector_store.similarity_search(query, k=3)

            context = ""

            for i, doc in enumerate(results, start=1):
                st.markdown(f"### Similar Case {i}")
                st.write(doc.page_content)
                context += doc.page_content + "\n"

            # Generate explanation
            prompt = f"""
            You are an e-commerce risk analyst.

            A user searched for: "{query}"

            Here are similar historical seller risk cases:

            {context}

            Explain:
            1. Why these sellers may be risky
            2. What patterns exist
            3. What business action should be taken

            Keep explanation simple and business-friendly.
            """

            report = genai_engine.generate_risk_report(prompt)

            st.markdown("## AI Risk Explanation")
            st.write(report)
        except Exception as e:
            st.error(f"Search error: {e}")
