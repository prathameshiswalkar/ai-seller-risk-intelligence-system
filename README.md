# AI Seller Risk Intelligence System

<div align="center">
An end-to-end AI platform for detecting, analyzing, and explaining seller risk in e-commerce marketplaces.
</div>

## Overview

The AI Seller Risk Intelligence System is a full-stack business intelligence platform built on the Brazilian E-Commerce Olist dataset. It combines machine learning, deep learning, generative AI, and vector memory into a single Streamlit application that helps marketplace analysts identify, investigate, and act on seller risk.

The platform goes beyond static dashboards by generating explainable insights from live seller metrics and historical risk patterns retrieved through a Retrieval-Augmented Generation (RAG) pipeline.

## Live Demo

[Streamlit App](https://ai-seller-risk-intelligence-system-4ne3qfvmuubmqvzgqmpdpa.streamlit.app/)

## System Architecture

```text
Olist Dataset (Raw)
  -> Data Processing with Pandas
  -> Seller Master Dataset
     -> ML Layer: XGBoost risk scoring
     -> DL Layer: BERT-based sentiment analysis
     -> GenAI Layer: Groq risk explanation
     -> RAG Memory Layer: FAISS / TF-IDF retrieval
  -> Streamlit Web Interface
```

## Features

### Executive Dashboard

- Marketplace overview with seller count, revenue, health, and risk indicators
- High-level business intelligence for fast executive review
- Platform capability summary across ML, DL, GenAI, and RAG layers

### Seller Risk Analyzer

- Custom seller metric input through Streamlit controls
- Real-time health score calculation
- AI-generated risk assessment using Groq
- Structured narrative output for diagnosis and next actions

### Review Sentiment Intelligence

- Review sentiment analysis in English and Portuguese
- `nlptown/bert-base-multilingual-uncased-sentiment` model support
- 1-5 star rating, sentiment class, and probability distribution
- Graceful fallback to a handcrafted multilingual rule-based analyzer when model weights are unavailable

### Seller Performance Dashboard

- Platform-wide KPI overview
- Risk segmentation across sellers
- Seller-level benchmark comparison
- Revenue vs. health scatter plot
- Top high-risk sellers table

### Risk Memory Explorer

- Natural-language search over historical seller risk records
- Similarity search using FAISS embeddings when available
- TF-IDF fallback retrieval when embeddings are unavailable
- AI-generated explanation of similar historical cases using Groq

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend / App | Streamlit 1.31 |
| Data Processing | Pandas 2.1, NumPy 1.26 |
| ML Model | XGBoost 2.0, Scikit-learn 1.3, Joblib |
| DL / Sentiment | PyTorch 2.1, Hugging Face Transformers 4.41 |
| Embeddings | Sentence Transformers 2.7 (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS-CPU 1.7, LangChain 0.1, LangChain Community |
| Generative AI | Groq SDK 1.1, LLaMA 3.1-8B-Instant |
| Visualization | Matplotlib 3.8 |
| Config / Secrets | python-dotenv 1.0, Streamlit Secrets |
| Dataset | Olist Brazilian E-Commerce Public Dataset |

## Project Structure

```text
ai-seller-risk-intelligence-system/
|- app/
|  |- main.py
|  |- bootstrap.py
|  \- pages/
|     |- 0_Executive_Dashboard.py
|     |- 1_Seller_Risk_Analyzer.py
|     |- 2_Sentiment_Intelligence.py
|     |- 3_Seller_Dashboard.py
|     \- 4_Risk_Memory_Explorer.py
|- src/
|  \- inference/
|     |- genai_engine.py
|     |- risk_engine.py
|     \- sentiment_engine.py
|- data/
|  \- processed/
|- models/
|  |- xgb_model.pkl
|  \- seller_memory_index/
|- notebook/
|- .streamlit/
|  \- config.toml
|- requirements.txt
|- runtime.txt
\- setup.py
```

## Getting Started

### Prerequisites

- Python 3.11
- A Groq API key
- Git

### Installation

```bash
git clone https://github.com/prathameshiswalkar/ai-seller-risk-intelligence-system.git
cd ai-seller-risk-intelligence-system

python -m venv venv
```

Activate the environment:

```bash
source venv/bin/activate
```

On Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_key_here
```

### Run the App

```bash
streamlit run app/main.py
```

The app will open at `http://localhost:8501`.

## Configuration

### Groq API Key

The app checks for `GROQ_API_KEY` in this order:

1. Environment variables
2. `.env` in the project root
3. Streamlit secrets

For Streamlit Cloud, add this to app secrets:

```toml
GROQ_API_KEY = "gsk_..."
```

## Streamlit Cloud Deployment

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from this repository.
4. Set the main file path to `app/main.py`.
5. Add `GROQ_API_KEY` in app secrets.
6. Deploy or reboot the app.

## Key Design Decisions

- Graceful degradation across AI components: the sentiment layer falls back to a multilingual rule-based analyzer when transformer weights are unavailable.
- FAISS with TF-IDF fallback: the memory explorer uses semantic embeddings when possible and switches to local lexical retrieval when needed.
- Dynamic path resolution: `bootstrap.py` ensures Streamlit multipage imports work reliably.
- Importlib-based page loading: inference modules are imported in a way that fits Streamlit's multipage execution model.

## Dataset

This project uses the [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

| File | Description |
| --- | --- |
| `olist_orders_dataset.csv` | Order lifecycle data |
| `olist_order_items_dataset.csv` | Item-level order details |
| `olist_order_reviews_dataset.csv` | Customer reviews used for sentiment |
| `olist_sellers_dataset.csv` | Seller location data |
| `olist_order_payments_dataset.csv` | Payment and revenue data |
| `olist_customers_dataset.csv` | Customer records |
| `olist_products_dataset.csv` | Product catalog |
| `olist_geolocation_dataset.csv` | Brazil geolocation mapping |
| `product_category_name_translation.csv` | Portuguese to English category translation |

The raw data is processed into seller-level datasets such as `seller_master.csv`, which powers the dashboards, risk logic, and memory explorer.

## Author

Prathamesh Iswalkar  
Post Graduate Program in Data Science & Machine Learning - Imarticus Learning

<div align="center">
<sub>Built with Python, Streamlit, XGBoost, BERT, FAISS, LangChain, and Groq</sub>
</div>
