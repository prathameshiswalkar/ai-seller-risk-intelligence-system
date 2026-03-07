import os
import sys

try:
    import google.genai as genai
except ImportError:
    print("WARNING: google.genai not found, trying google.generativeai")
    import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize client with API key
api_key = os.getenv("GEMINI_API_KEY")
if api_key is None:
    print("WARNING: GEMINI_API_KEY environment variable not set")

try:
    client = genai.Client(api_key=api_key)
except AttributeError:
    # Fallback for older google.generativeai
    genai.configure(api_key=api_key)
    client = None

# Absolute Path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
INDEX_PATH = os.path.join(BASE_DIR, "models", "seller_memory_index")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector store with fallback
vector_store = None
try:
    if os.path.exists(INDEX_PATH):
        vector_store = FAISS.load_local(
            INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        print(f"WARNING: FAISS index not found at {INDEX_PATH}")
except Exception as e:
    print(f"WARNING: Could not load FAISS index: {e}")

def generate_risk_report(prompt):
    if client is not None:
        # New google.genai API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
    else:
        # Fallback to old google.generativeai API
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
    return response.text