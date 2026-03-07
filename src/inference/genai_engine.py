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


# ---------------------------------------------------
# Paths
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_PATH = os.path.join(BASE_DIR, "models", "seller_memory_index")


# ---------------------------------------------------
# Gemini Client (cached)
# ---------------------------------------------------

@st.cache_resource
def load_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("WARNING: GEMINI_API_KEY not found")
        return None

    try:
        if NEW_API:
            return genai.Client(api_key=api_key)
        else:
            genai.configure(api_key=api_key)
            return None
    except Exception as e:
        print(f"Gemini client error: {e}")
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

    if not os.path.exists(INDEX_PATH):
        print(f"WARNING: FAISS index not found at {INDEX_PATH}")
        return None

    try:
        return FAISS.load_local(
            INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"WARNING: Could not load FAISS index: {e}")
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
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text

    except Exception as e:
        return f"Gemini error: {str(e)}"