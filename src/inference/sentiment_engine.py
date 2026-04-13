import math
import re
import unicodedata
import warnings

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
TOKEN_PATTERN = re.compile(r"\b[\w']+\b")
POSITIVE_HINTS = {
    "good", "great", "excellent", "love", "amazing", "fast", "perfect", "happy",
    "best", "awesome", "fantastic", "brilliant", "satisfied", "smooth", "reliable",
    "recomendo", "maravilhoso", "perfeito", "gostei",
    "bom", "otimo", "ótimo", "excelente", "rapido", "rápido", "adorei", "parabens",
}
NEGATIVE_HINTS = {
    "bad", "terrible", "awful", "late", "broken", "poor", "worst", "refund",
    "damaged", "disappointing", "horrible", "useless", "fake", "delay", "delayed",
    "pessimo", "danificado", "quebrado", "demorou", "decepcionante",
    "ruim", "atrasado", "atrasou", "horrivel", "horrível", "péssimo", "problema",
}
POSITIVE_PHRASES = {
    "on time": 1.0,
    "before the promised date": 1.4,
    "well packaged": 1.0,
    "high quality": 1.5,
    "works perfectly": 1.6,
    "very happy": 1.6,
    "worth the price": 1.2,
    "fast delivery": 1.5,
    "great service": 1.5,
    "highly recommend": 1.8,
    "entrega rapida": 1.5,
    "chegou rapido": 1.5,
    "chegou antes": 1.7,
}
NEGATIVE_PHRASES = {
    "arrived late": 1.8,
    "very late": 1.6,
    "poor quality": 1.7,
    "not worth": 1.6,
    "waste of money": 2.0,
    "broken product": 2.0,
    "bad service": 1.6,
    "no response": 1.4,
    "damaged product": 2.0,
    "customer support was terrible": 2.0,
    "entrega atrasada": 1.8,
    "produto ruim": 1.8,
    "embalagem danificada": 1.7,
}
INTENSIFIERS = {
    "very", "really", "extremely", "super", "too", "so", "highly",
    "muito", "bem", "bastante",
}
NEGATIONS = {
    "not", "never", "no", "hardly", "barely", "without",
    "nao", "nem",
}
CONTRAST_PATTERN = re.compile(r"\b(?:but|however|though|although|yet|mas|porem)\b")


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


def _strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def _normalize_text(text: str) -> str:
    normalized = _strip_accents(text.lower().strip())
    return re.sub(r"\s+", " ", normalized)


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def _apply_phrase_weights(text: str, phrase_map: dict[str, float]) -> float:
    score = 0.0
    for phrase, weight in phrase_map.items():
        if phrase in text:
            score += weight
    return score


def _score_tokens(tokens: list[str]) -> tuple[float, float]:
    positive_score = 0.0
    negative_score = 0.0

    for index, token in enumerate(tokens):
        window = tokens[max(0, index - 2):index]
        multiplier = 1.0 + (0.35 if any(word in INTENSIFIERS for word in window) else 0.0)
        is_negated = any(word in NEGATIONS for word in window)

        if token in POSITIVE_HINTS:
            if is_negated:
                negative_score += 1.1 * multiplier
            else:
                positive_score += 1.0 * multiplier
        elif token in NEGATIVE_HINTS:
            if is_negated:
                positive_score += 0.9 * multiplier
            else:
                negative_score += 1.1 * multiplier

    return positive_score, negative_score


def _score_segment(segment: str) -> tuple[float, float]:
    positive_score = _apply_phrase_weights(segment, POSITIVE_PHRASES)
    negative_score = _apply_phrase_weights(segment, NEGATIVE_PHRASES)

    token_positive, token_negative = _score_tokens(_tokenize(segment))
    positive_score += token_positive
    negative_score += token_negative

    if "!" in segment:
        if positive_score > negative_score:
            positive_score += 0.2
        elif negative_score > positive_score:
            negative_score += 0.2

    return positive_score, negative_score


def _build_fallback_distribution(sentiment_score: float) -> tuple[int, list[float]]:
    if sentiment_score >= 1.4:
        return 5, [0.01, 0.03, 0.06, 0.18, 0.72]
    if sentiment_score >= 0.45:
        return 4, [0.03, 0.07, 0.15, 0.50, 0.25]
    if sentiment_score <= -1.4:
        return 1, [0.72, 0.18, 0.06, 0.03, 0.01]
    if sentiment_score <= -0.45:
        return 2, [0.25, 0.50, 0.15, 0.07, 0.03]
    return 3, [0.08, 0.16, 0.52, 0.16, 0.08]


def _heuristic_sentiment(text: str):
    normalized = _normalize_text(text)

    if not normalized:
        rating, probabilities = 3, [0.08, 0.16, 0.52, 0.16, 0.08]
    else:
        segments = [segment.strip() for segment in CONTRAST_PATTERN.split(normalized) if segment.strip()]
        positive_score = 0.0
        negative_score = 0.0

        for index, segment in enumerate(segments):
            pos_score, neg_score = _score_segment(segment)
            weight = 1.25 if len(segments) > 1 and index == len(segments) - 1 else 1.0
            positive_score += pos_score * weight
            negative_score += neg_score * weight

        token_count = max(len(_tokenize(normalized)), 1)
        sentiment_score = (positive_score - negative_score) / math.sqrt(token_count)
        rating, probabilities = _build_fallback_distribution(sentiment_score)

    if rating <= 2:
        sentiment = "negative"
    elif rating >= 4:
        sentiment = "positive"
    else:
        sentiment = "neutral"

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
