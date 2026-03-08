import os
import pathlib
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------------------------------------
# Project root
# ------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]

INDEX_PATH = BASE_DIR / "models" / "seller_memory_index"

print("BASE_DIR:", BASE_DIR)
print("INDEX_PATH:", INDEX_PATH)
print("INDEX EXISTS:", INDEX_PATH.exists())

# ------------------------------------------------
# Embedding model
# ------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()

# ------------------------------------------------
# Load FAISS
# ------------------------------------------------

@st.cache_resource
def load_vector_store():

    if not INDEX_PATH.exists():
        print("FAISS folder not found:", INDEX_PATH)
        return None

    faiss_file = INDEX_PATH / "index.faiss"
    pkl_file = INDEX_PATH / "index.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        print("FAISS files missing")
        return None

    try:

        vector_store = FAISS.load_local(
            str(INDEX_PATH),
            embedding_model,
            allow_dangerous_deserialization=True
        )

        print("FAISS loaded successfully")

        return vector_store

    except Exception as e:

        print("FAISS load error:", e)

        return None


vector_store = load_vector_store()