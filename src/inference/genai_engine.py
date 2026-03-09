import pathlib
import pandas as pd
import os
import warnings

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import google.generativeai as genai

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "seller_master.csv"

INDEX_PATH = BASE_DIR / "models" / "seller_memory_index"

print("BASE_DIR:", BASE_DIR)
print("DATA_PATH:", DATA_PATH)

vector_store = None


# --------------------------------------------------
# Build RAG Vector Store From Dataset
# --------------------------------------------------

def build_vector_store():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    documents = []

    for row in df.itertuples():

        negative_rate = getattr(row, "negative_rate", 0)

        text = f"""
        Seller ID: {row.seller_id}
        Revenue: {row.total_revenue}
        Late Delivery Rate: {row.late_delivery_rate}
        Negative Review Rate: {negative_rate}
        Seller Health Index: {row.seller_health_index_v2}
        """

        documents.append(Document(page_content=text))

    print("Documents created:", len(documents))

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(documents, embedding_model)

    print("Vector store created successfully")

    return vector_store


# --------------------------------------------------
# Initialize Vector Store
# --------------------------------------------------

import streamlit as st

@st.cache_resource
def load_vector_store():
    try:
        return build_vector_store()
    except Exception as e:
        print("RAG initialization error:", e)
        return None

vector_store = load_vector_store()


# --------------------------------------------------
# Gemini Risk Report Generator
# --------------------------------------------------

def generate_risk_report(prompt: str):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "GEMINI_API_KEY environment variable not set."

    try:

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:

        return f"AI generation error: {str(e)}"