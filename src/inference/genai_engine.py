import pathlib
import pandas as pd
import os
import warnings

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import google.generativeai as genai

# Suppress huggingface_hub deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]

INDEX_PATH = BASE_DIR / "models" / "seller_memory_index"
DATA_PATH = BASE_DIR / "data" / "processed" / "seller_master.csv"

print("BASE_DIR:", BASE_DIR)
print("INDEX_PATH:", INDEX_PATH)
print("INDEX EXISTS:", INDEX_PATH.exists())

vector_store = None

try:

    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------
    # CASE 1: Load existing FAISS index
    # --------------------------------------------------

    if INDEX_PATH.exists():

        print("Loading existing FAISS index...")

        vector_store = FAISS.load_local(
            str(INDEX_PATH),
            embedding_model,
            allow_dangerous_deserialization=True
        )

        print("FAISS loaded successfully")

    # --------------------------------------------------
    # CASE 2: Build FAISS index automatically
    # --------------------------------------------------

    else:

        print("FAISS index not found. Building new index...")

        df = pd.read_csv(DATA_PATH)

        documents = []

        for _, row in df.iterrows():

            negative_rate = row["negative_rate"] if "negative_rate" in df.columns else 0

            text = f"""
            Seller ID: {row['seller_id']}
            Revenue: {row['total_revenue']}
            Late Delivery Rate: {row['late_delivery_rate']}
            Negative Review Rate: {negative_rate}
            Seller Health Index: {row['seller_health_index_v2']}
            """

            documents.append(Document(page_content=text))

        print("Total documents:", len(documents))

        vector_store = FAISS.from_documents(documents, embedding_model)

        INDEX_PATH.mkdir(parents=True, exist_ok=True)

        vector_store.save_local(str(INDEX_PATH))

        print("FAISS index built and saved successfully")

except Exception as e:

    print("FAISS loading error:", e)
    vector_store = None


# --------------------------------------------------
# Risk Report Generator (used by Seller Risk Analyzer)
# --------------------------------------------------

def generate_risk_report(prompt):

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.0-pro")
    response = model.generate_content(prompt)
    return response.text