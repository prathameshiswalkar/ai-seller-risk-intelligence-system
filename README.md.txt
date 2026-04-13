# AI Seller Risk Intelligence System

**Live Application:**
[https://ai-seller-risk-intelligence-system-4ne3qfvmuubmqvzgqmpdpa.streamlit.app/](https://ai-seller-risk-intelligence-system-4ne3qfvmuubmqvzgqmpdpa.streamlit.app/)

## Overview

The **AI Seller Risk Intelligence System** is an interactive analytics platform designed to identify and investigate high-risk sellers in an e-commerce marketplace. The application combines **machine learning, sentiment analysis, vector search (RAG), and generative AI** to analyze seller performance, detect potential risks, and provide explainable insights for decision-making.

The system helps analysts quickly evaluate seller behavior, monitor operational performance, and investigate historical risk patterns.

---

# How the Application Works

## 1. Data Processing

The system uses a processed seller dataset containing operational metrics such as:

* Seller ID
* Total Revenue
* Late Delivery Rate
* Negative Review Rate
* Seller Health Index

This data represents seller performance indicators commonly used in marketplace risk analysis.

The dataset is loaded and processed using **Pandas**, preparing it for machine learning analysis and vector search.

---

## 2. Executive Dashboard

The **Executive Dashboard** provides a high-level overview of the marketplace.

Key insights displayed include:

* Total seller count
* Average seller performance metrics
* Marketplace health indicators
* Operational risk distribution

This dashboard helps decision-makers quickly understand the overall marketplace risk environment.

---

## 3. Seller Risk Analyzer (Machine Learning)

The **Seller Risk Analyzer** uses a trained machine learning model to evaluate whether a seller is likely to be risky.

The model analyzes seller metrics such as:

* Delivery performance
* Customer review patterns
* Revenue behavior
* Seller health index

Based on these indicators, the model predicts potential risk levels and highlights sellers that may require investigation.

---

## 4. Sentiment Intelligence

Customer feedback is an important signal in risk detection.

The **Sentiment Intelligence module** analyzes seller review data to determine:

* Positive sentiment
* Neutral sentiment
* Negative sentiment

This allows analysts to detect early warning signals such as increasing customer dissatisfaction.

---

## 5. Seller Performance Dashboard

This module visualizes seller operational metrics to provide deeper insights into performance patterns.

Analysts can observe:

* Delivery performance trends
* Revenue comparisons
* Customer feedback signals
* Seller health indicators

The dashboard helps identify sellers that deviate from normal operational behavior.

---

## 6. Risk Memory Explorer (Retrieval Augmented Generation)

The **Risk Memory Explorer** is an AI investigation tool that retrieves similar historical seller risk cases.

### Step 1: Vector Embedding

Seller records are converted into textual documents containing seller metrics.

Example document:

Seller ID: S104
Revenue: 23000
Late Delivery Rate: 0.42
Negative Review Rate: 0.37
Seller Health Index: 0.48

These documents are converted into vector embeddings using a **HuggingFace sentence transformer model**.

---

### Step 2: FAISS Vector Search

The embeddings are stored in a **FAISS vector database**.

When a user searches for a risk pattern, the system performs **similarity search** to retrieve the most relevant historical seller cases.

---

### Step 3: AI Explanation with Groq

The retrieved cases are passed to **Groq**, which generates an explanation describing:

* Risk patterns in the sellers
* Possible causes of operational issues
* Recommended actions for investigation

This creates an **explainable AI system** that helps analysts understand the reasoning behind risk detection.

---

# System Architecture

```
Seller Dataset
      ↓
Data Processing (Pandas)
      ↓
Machine Learning Risk Model
      ↓
Vector Embeddings (Sentence Transformers)
      ↓
FAISS Vector Database
      ↓
Similarity Search
      ↓
Groq AI Explanation
      ↓
Interactive Streamlit Dashboard
```

---

# Technologies Used

* **Python**
* **Streamlit**
* **Pandas & NumPy**
* **Scikit-learn / XGBoost**
* **FAISS Vector Database**
* **HuggingFace Sentence Transformers**
* **LangChain**
* **Groq API**

---

# Key Capabilities

* AI-powered seller risk detection
* Explainable AI insights for investigation
* Sentiment analysis of customer feedback
* Retrieval of historical risk patterns using RAG
* Interactive analytics dashboards

---

# Author

**Prathamesh Iswalkar**
Aspiring Business & Data Analyst | AI & Data Analytics Enthusiast
