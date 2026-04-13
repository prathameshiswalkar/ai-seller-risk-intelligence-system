import warnings

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
POSITIVE_HINTS = {
    "good", "great", "excellent", "love", "amazing", "fast", "perfect", "happy",
    "bom", "otimo", "ótimo", "excelente", "rapido", "rápido", "adorei", "parabens",
}
NEGATIVE_HINTS = {
    "bad", "terrible", "awful", "late", "broken", "poor", "worst", "refund",
    "ruim", "atrasado", "atrasou", "horrivel", "horrível", "péssimo", "problema",
}


@st.cache_resource
def load_sentiment_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device
    except Exception:
        return None, None, None


def _heuristic_sentiment(text: str):
    normalized = text.lower()
    positive_hits = sum(word in normalized for word in POSITIVE_HINTS)
    negative_hits = sum(word in normalized for word in NEGATIVE_HINTS)

    if positive_hits > negative_hits:
        rating = 4 if positive_hits == negative_hits + 1 else 5
    elif negative_hits > positive_hits:
        rating = 2 if negative_hits == positive_hits + 1 else 1
    else:
        rating = 3

    if rating <= 2:
        sentiment = "negative"
        probabilities = [0.55, 0.25, 0.10, 0.06, 0.04]
    elif rating >= 4:
        sentiment = "positive"
        probabilities = [0.04, 0.06, 0.10, 0.25, 0.55]
    else:
        sentiment = "neutral"
        probabilities = [0.10, 0.20, 0.40, 0.20, 0.10]

    return {
        "sentiment": sentiment,
        "rating": rating,
        "confidence": max(probabilities),
        "probabilities": probabilities,
        "used_fallback": True,
    }


def analyze_sentiment(text):
    tokenizer, model, device = load_sentiment_model()

    if tokenizer is None or model is None or device is None:
        return _heuristic_sentiment(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    rating = torch.argmax(probs, dim=1).item() + 1
    confidence = torch.max(probs).item()

    if rating <= 2:
        sentiment = "negative"
    elif rating >= 4:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "rating": rating,
        "confidence": confidence,
        "probabilities": probs.squeeze().tolist(),
        "used_fallback": False,
    }
