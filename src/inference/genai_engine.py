import pathlib
import pandas as pd
import os
import warnings

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import google.generativeai as genai

# Suppress huggingface warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]

INDEX_PATH = BASE_DIR / "models" / "seller_memory_index"
DATA_PATH = BASE_DIR / "data" / "processed" / "seller_master.csv"

print("BASE_DIR:", BASE_DIR)
print("INDEX_PATH:", INDEX_PATH)

vector_store = None


def build_faiss_index():
    """Build FAISS index from seller dataset"""

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

    vs = FAISS.from_documents(documents, embedding_model)

    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_PATH))

    print("FAISS index built successfully")

    return vs


try:

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    faiss_file = INDEX_PATH / "index.faiss"

    # --------------------------------------------------
    # Load existing FAISS index
    # --------------------------------------------------

    if faiss_file.exists():

        print("Loading existing FAISS index...")

        vector_store = FAISS.load_local(
            str(INDEX_PATH),
            embedding_model,
            allow_dangerous_deserialization=True
        )

        print("FAISS loaded successfully")

    # --------------------------------------------------
    # Build index automatically
    # --------------------------------------------------

    else:

        print("FAISS index not found. Building new index...")

        vector_store = build_faiss_index()

except Exception as e:

    print("FAISS error:", e)
    vector_store = None


# --------------------------------------------------
# Gemini Risk Report Generator
# --------------------------------------------------

def generate_risk_report(prompt: str):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "GEMINI_API_KEY environment variable not set."

    try:

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:

        return f"AI generation error: {str(e)}"