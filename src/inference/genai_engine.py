import os
import streamlit as st

try:
    import google.genai as genai
    NEW_API = True
except ImportError:
    import google.generativeai as genai
    NEW_API = False

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pathlib


# ---------------------------------------------------
# Paths (FIXED FOR STREAMLIT CLOUD)
# ---------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]

INDEX_PATH = BASE_DIR / "models" / "seller_memory_index"

print("BASE_DIR:", BASE_DIR)
print("INDEX_PATH:", INDEX_PATH)
print("INDEX EXISTS:", INDEX_PATH.exists())


# ---------------------------------------------------
# Gemini Client
# ---------------------------------------------------

@st.cache_resource
def load_gemini_client():

    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

    if not api_key:
        st.warning("GEMINI_API_KEY not configured. GenAI features disabled.")
        return None

    try:
        if NEW_API:
            return genai.Client(api_key=api_key)
        else:
            genai.configure(api_key=api_key)
            return genai
    except Exception as e:
        st.error(f"Gemini client error: {e}")
        return None


client = load_gemini_client()


# ---------------------------------------------------
# Embedding Model
# ---------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ---------------------------------------------------
# FAISS Vector Store
# ---------------------------------------------------

@st.cache_resource
def load_vector_store():

    if not INDEX_PATH.exists():
        print(f"Vector index folder missing: {INDEX_PATH}")
        return None

    faiss_file = INDEX_PATH / "index.faiss"
    pkl_file = INDEX_PATH / "index.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        print("FAISS index files missing.")
        return None

    try:
        vector_store = FAISS.load_local(
            str(INDEX_PATH),
            embedding_model,
            allow_dangerous_deserialization=True
        )

        print("FAISS vector store loaded successfully")
        return vector_store

    except Exception as e:
        print(f"FAISS load error: {e}")
        return None


vector_store = load_vector_store()


# ---------------------------------------------------
# Risk Report Generator
# ---------------------------------------------------

def generate_risk_report(prompt):

    if client is None:
        return "Gemini API not configured."

    try:

        if NEW_API:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            return response.text

        else:

            model = client.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)

            return response.text

    except Exception as e:

        return f"Gemini error: {str(e)}"