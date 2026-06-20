"""
========================================================
Enhanced BERT (Pulk17/Fake-News-Detection)
NeuroLex Fake News Detection — Production Version
========================================================
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from loguru import logger
from typing import Dict, Optional

# Global references
_model: Optional[AutoModelForSequenceClassification] = None
_tokenizer: Optional[AutoTokenizer] = None
_device: Optional[torch.device] = None
DEVICE = None

# Model priority
MODEL_PRIORITY = [
    "my_local_model",                # optional custom fine-tuned checkpoint
    "Pulk17/Fake-News-Detection",    # Enhanced BERT fake news detection (Hugging Face)
    "bert-base-uncased"              # fallback
]


def load_bert_model() -> bool:
    """Load Enhanced BERT model with corrected label mapping."""
    global _model, _tokenizer, _device, DEVICE

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = _device
    logger.info(f"🖥️ Using device: {_device}")

    for model_name in MODEL_PRIORITY:
        try:
            logger.info(f"🧠 Loading model: {model_name} ...")
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Correct the label mapping (LABEL_0 → FAKE, LABEL_1 → REAL)
            _model.config.id2label = {0: "FAKE", 1: "REAL"}
            _model.config.label2id = {"FAKE": 0, "REAL": 1}

            _model.to(_device)
            _model.eval()

            logger.info(f"✅ Model '{model_name}' loaded successfully on {_device}.")
            logger.info(f"Labels remapped: {_model.config.id2label}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to load model '{model_name}': {e}")
            continue

    logger.error("❌ No model could be loaded from MODEL_PRIORITY.")
    return False


def predict_fake_news(text: str, max_length: int = 512) -> Dict:
    """Predict whether the given text represents real or fake news."""
    if _model is None or _tokenizer is None:
        raise RuntimeError("❌ Model not loaded — call load_bert_model() first.")

    inputs = _tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_class = int(torch.argmax(logits, dim=1).item())
    label = _model.config.id2label[predicted_class]
    confidence = float(probs[predicted_class])
    probs_dict = {label_name: float(probs[i]) for i, label_name in _model.config.id2label.items()}

    logger.info(f"Prediction: {label} (Confidence: {confidence:.3f}) — All: {probs_dict}")
    return {"label": label, "confidence": confidence, "probabilities": probs_dict}


def get_model() -> Optional[AutoModelForSequenceClassification]:
    return _model


def get_tokenizer() -> Optional[AutoTokenizer]:
    return _tokenizer


def get_device() -> Optional[torch.device]:
    return _device


if __name__ == "__main__":
    logger.add("fake_news.log")
    logger.info("🚀 Testing Enhanced BERT fake news classifier...")

    if load_bert_model():
        samples = [
            "NASA confirms water found on Mars.",
            "Scientists discovered a cure for cancer but it was hidden by governments."
        ]
        for text in samples:
            result = predict_fake_news(text)
            print(f"\nInput: {text}\nOutput: {result}")
    else:
        print("❌ Model loading failed.")
