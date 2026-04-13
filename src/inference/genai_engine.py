import os
import pathlib
import re
import warnings
from dataclasses import dataclass

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "seller_master.csv"
INDEX_PATH = BASE_DIR / "models" / "seller_memory_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv(BASE_DIR / ".env")


def _get_groq_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key

    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def _row_to_document(row) -> Document:
    negative_rate = getattr(row, "negative_rate", 0)
    risk_level = getattr(row, "risk_level", "UNKNOWN")
    text = (
        f"Seller ID: {row.seller_id}\n"
        f"Revenue: {row.total_revenue}\n"
        f"Late Delivery Rate: {row.late_delivery_rate}\n"
        f"Negative Review Rate: {negative_rate}\n"
        f"Seller Health Index: {row.seller_health_index_v2}\n"
        f"Risk Level: {risk_level}"
    )
    return Document(page_content=text)


def _load_documents() -> list[Document]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    return [_row_to_document(row) for row in df.itertuples()]


@dataclass
class LocalRiskMemoryStore:
    documents: list[Document]
    vectorizer: TfidfVectorizer
    matrix: object
    backend: str = "tfidf"

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        query = (query or "").strip()
        if not query:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = linear_kernel(query_vector, self.matrix).flatten()

        numeric_tokens = [float(token) for token in re.findall(r"\d+(?:\.\d+)?", query)]
        if numeric_tokens:
            for index, document in enumerate(self.documents):
                doc_numbers = [
                    float(token)
                    for token in re.findall(r"\d+(?:\.\d+)?", document.page_content)
                ]
                if not doc_numbers:
                    continue
                distance_bonus = sum(
                    1 / (1 + abs(query_value - doc_value))
                    for query_value in numeric_tokens
                    for doc_value in doc_numbers[:4]
                )
                scores[index] += distance_bonus * 0.05

        top_indices = scores.argsort()[::-1][:k]
        return [self.documents[index] for index in top_indices if scores[index] > 0]


def _build_local_memory_store(documents: list[Document]) -> LocalRiskMemoryStore:
    texts = [document.page_content for document in documents]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return LocalRiskMemoryStore(documents=documents, vectorizer=vectorizer, matrix=matrix)


def _build_huggingface_vector_store(documents: list[Document]):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"local_files_only": True},
    )
    vector_store = FAISS.from_documents(documents, embedding_model)
    setattr(vector_store, "backend", "faiss")
    return vector_store


def build_vector_store():
    documents = _load_documents()

    try:
        return _build_huggingface_vector_store(documents)
    except Exception:
        return _build_local_memory_store(documents)


@st.cache_resource
def load_vector_store():
    try:
        return build_vector_store()
    except Exception:
        return None


_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = load_vector_store()
    return _vector_store


def generate_risk_report(prompt: str):
    api_key = _get_groq_api_key()

    if not api_key:
        return "GROQ_API_KEY environment variable not set."

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert e-commerce risk analyst. Analyze seller "
                        "risk patterns and provide clear, structured insights."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"AI generation error: {exc}"
