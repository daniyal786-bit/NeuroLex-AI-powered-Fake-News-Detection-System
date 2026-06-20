import re
from loguru import logger

from core.config import settings
from .model_loader import load_bert_model, get_model, get_tokenizer, DEVICE

# Import torch lazily within functions via model_loader to avoid DLL errors at import time

# Safe import for SHAP
try:
    import shap
except (ImportError, ModuleNotFoundError):
    shap = None

# Safe import for numpy
try:
    import numpy as np
except (ImportError, ModuleNotFoundError):
    np = None

# --- Heuristic-Based Prediction ---

def predict_fake_real_heuristic(text: str):
    """A simple heuristic-based fake news predictor."""
    text_lower = (text or "").lower()
    words = set(re.findall(r"\w+", text_lower))
    score = sum(1 for w in settings.sensational_words if w in words)
    
    if score > 0:
        confidence = min(0.5 + 0.15 * score, 0.99)
        return {"label": "fake", "confidence": round(confidence, 2)}
    else:
        return {"label": "real", "confidence": 0.85}

# --- BERT-Based Prediction ---

def predict_fake_real_bert(text: str):
    """Predicts fake/real using a BERT model, with a fallback to heuristics."""
    # Ensure model is available
    if not load_bert_model():
        return predict_fake_real_heuristic(text)

    text = (text or "").strip()
    if not text:
        return {"label": "real", "confidence": 0.85}

    tokenizer = get_tokenizer()
    model = get_model()

    if not tokenizer or not model:
        return predict_fake_real_heuristic(text)

    try:
        # Import torch inside the function to avoid import-time failures
        import torch
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()
            pred_idx = int(torch.argmax(probs).item())
            label = "fake" if pred_idx == 1 else "real"
            confidence = float(probs.max().item())
        return {"label": label, "confidence": round(confidence, 2)}
    except Exception as e:
        logger.error(f"BERT prediction failed: {e}")
        return predict_fake_real_heuristic(text)  # Fallback on error

# --- Explanation Functions ---

def explain_text_heuristic(text: str):
    """Generates a simple keyword-based explanation."""
    words = re.findall(r"\w+", (text or "").lower())
    explanations = []
    seen = set()
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        if w in settings.sensational_words:
            explanations.append({"word": w, "weight": 0.8, "reason": "sensational keyword"})
    return explanations[:8]

def explain_model_tokens(text: str, top_k: int = 12):
    """Explains a prediction using model attention scores as an approximation."""
    if not np or not load_bert_model():
        return []

    tokenizer = get_tokenizer()
    model = get_model()
    if not tokenizer or not model:
        return []

    try:
        import torch
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs, output_attentions=True)
        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            return []

        last_layer_attention = attentions[-1]
        mean_heads = last_layer_attention.mean(dim=1)
        attn_cls = mean_heads[0, 0, :].cpu().numpy()
        
        token_ids = inputs["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Simple word merging from tokens
        words, weights = _merge_tokens_to_words(tokens, attn_cls, tokenizer)
        if not weights:
            return []

        arr = np.array(weights)
        norm_weights = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        
        explanation = [{
            "word": w,
            "weight": round(float(s), 3),
            "reason": "attention_score"
        } for w, s in zip(words, norm_weights)]
        
        return sorted(explanation, key=lambda x: x["weight"], reverse=True)[:top_k]
    except Exception as e:
        logger.error(f"Token explanation failed: {e}")
        return []

def _merge_tokens_to_words(tokens, scores, tokenizer):
    words = []
    weights = []
    current_word = ""
    current_weight = 0.0
    current_count = 0

    for token, score in zip(tokens, scores):
        if token in (tokenizer.cls_token, tokenizer.sep_token):
            continue

        is_subword = token.startswith("##")
        clean_token = token.replace("##", "")

        if not is_subword and current_word:
            words.append(current_word)
            weights.append(current_weight / max(1, current_count))
            current_word = clean_token
            current_weight = float(score)
            current_count = 1
        else:
            current_word += clean_token
            current_weight += float(score)
            current_count += 1
            
    if current_word:
        words.append(current_word)
        weights.append(current_weight / max(1, current_count))
        
    return words, weights
