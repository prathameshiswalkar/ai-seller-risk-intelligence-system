import sys
import os
import streamlit as st

# ---------------------------------------------------
# Ensure project root is on Python path
# ---------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------
# Import vector store from inference engine
# ---------------------------------------------------

vector_store = None
import_error = None

try:
    from src.inference.genai_engine import vector_store
except Exception as e:
    vector_store = None
    import_error = str(e)

# ---------------------------------------------------
# UI
# ---------------------------------------------------

st.title("Historical Risk Memory")

st.caption("Search similar historical seller risk cases using vector memory")

query = st.text_input("Search Similar Risk Cases")

# ---------------------------------------------------
# Debug info (remove later if you want)
# ---------------------------------------------------

with st.expander("System Debug Info"):
    st.write("Project Root:", PROJECT_ROOT)
    st.write("Vector store loaded:", vector_store is not None)
    if import_error:
        st.write("Import error:", import_error)

# ---------------------------------------------------
# Search
# ---------------------------------------------------

if st.button("Search"):

    if vector_store is None:
        st.error("Vector store not available. Please check configuration.")
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
            st.error(f"Search error: {str(e)}")