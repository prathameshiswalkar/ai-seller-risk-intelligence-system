import os
import streamlit as st

# try either of the two GenAI packages; fall back gracefully if neither
try:
    import google.genai as genai
    NEW_API = True
except ImportError:
    try:
        import google.generativeai as genai
        NEW_API = False
    except ImportError:
        genai = None
        NEW_API = False

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pathlib


# ---------------------------------------------------
# Paths (project root → models/seller_memory_index)
# ---------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
INDEX_PATH = BASE_DIR / "models" / "seller_memory_index"

print("BASE_DIR:", BASE_DIR)
print("INDEX_PATH:", INDEX_PATH)
print("INDEX EXISTS:", INDEX_PATH.exists())


# ---------------------------------------------------
# Gemini client
# ---------------------------------------------------

@st.cache_resource

def load_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    if not api_key or genai is None:
        st.warning("Gemini API not configured or library missing; "
                   "GenAI features disabled.")
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
# Embeddings
# ---------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ---------------------------------------------------
# FAISS vector store
# ---------------------------------------------------
@st.cache_resource
def load_vector_store():
    if not INDEX_PATH.exists():
        st.error(f"Vector‑index directory does not exist: {INDEX_PATH}")
        return None

    faiss_file = INDEX_PATH / "index.faiss"
    pkl_file   = INDEX_PATH / "index.pkl"
    if not faiss_file.exists() or not pkl_file.exists():
        st.error(
            "FAISS index files missing – run the script that builds the index "
            "and place the resulting `index.faiss`/`index.pkl` under "
            "`models/seller_memory_index`."
        )
        return None

    try:
        vs = FAISS.load_local(
            str(INDEX_PATH),
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        st.success("FAISS vector store loaded successfully")
        return vs
    except Exception as e:
        st.error(f"FAISS load error: {e}")
        return None


vector_store = load_vector_store()


# ---------------------------------------------------
# risk‑report helper
# ---------------------------------------------------
def generate_risk_report(prompt: str) -> str:
    if client is None:
        return "Gemini API not configured."
    try:
        if NEW_API:
            resp = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
        else:
            model = client.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Gemini error: {e}"