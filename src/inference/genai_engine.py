import pathlib
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]

INDEX_PATH = BASE_DIR / "models" / "seller_memory_index"

print("BASE_DIR:", BASE_DIR)
print("INDEX_PATH:", INDEX_PATH)
print("INDEX EXISTS:", INDEX_PATH.exists())

vector_store = None

try:

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        str(INDEX_PATH),
        embedding_model,
        allow_dangerous_deserialization=True
    )

    print("FAISS loaded successfully")

except Exception as e:

    print("FAISS loading error:", e)
    vector_store = None