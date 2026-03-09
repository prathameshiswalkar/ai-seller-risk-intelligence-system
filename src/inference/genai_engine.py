import pathlib
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

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

    # --------------------------------------------------
    # Load Embedding Model
    # --------------------------------------------------

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------
    # CASE 1: FAISS index already exists → load it
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
    # CASE 2: FAISS index missing → build automatically
    # --------------------------------------------------

    else:

        print("FAISS index not found. Building new index...")

        df = pd.read_csv(DATA_PATH)

        documents = []

        for _, row in df.iterrows():

            text = f"""
            Seller ID: {row['seller_id']}
            Revenue: {row['total_revenue']}
            Late Delivery Rate: {row['late_delivery_rate']}
            Negative Review Rate: {row['negative_review_rate']}
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