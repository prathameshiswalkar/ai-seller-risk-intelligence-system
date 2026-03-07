from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def analyze_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

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
        "probabilities": probs.squeeze().tolist()
    }