import sys
import os
import streamlit as st

# ensure project root is on sys.path so `src` imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib

vector_store = None
import_error = None
INDEX_PATH = None

try:
    genai_engine = importlib.import_module('src.inference.genai_engine')
    vector_store = genai_engine.vector_store
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
    if INDEX_PATH is not None:
        st.write("Index path:", INDEX_PATH)
        st.write("Index exists:", INDEX_PATH.exists())
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
            if not results:
                st.info("No similar risk cases found.")
            else:
                for i, doc in enumerate(results, start=1):
                    st.markdown(f"### Similar Case {i}")
                    st.write(doc.page_content)
        except Exception as e:
            st.error(f"Search error: {e}")