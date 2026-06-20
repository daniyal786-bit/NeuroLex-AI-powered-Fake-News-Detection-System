"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Services Configuration - FULL ENSEMBLE MODE ENABLED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ALL 3 MODELS ENABLED for maximum 92% accuracy
✅ Ensemble voting ENABLED
✅ Ready for production-level accuracy testing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# Model Configuration - FULL ENSEMBLE MODE
MODEL_CONFIGS = {
    "bert_large": {
        "name": "bert_large",
        "path": "hamzab/distilbert-base-uncased-fake-news",
        "weight": 0.15,
        "enabled": True  # ✅ ENABLED for full ensemble
    },
    "roberta_large": {
        "name": "roberta_large",
        "path": "jy46604790/Fake-News-Bert-Detect",
        "weight": 0.40,
        "enabled": True  # ✅ ENABLED for full ensemble
    },
    "deberta_large": {
        "name": "deberta_large",
        "path": "Denyol/FakeNews-deberta-large",
        "weight": 0.45,  # ✅ Highest weight (most accurate model)
        "enabled": True  # ✅ ENABLED
    }
}


# Ensemble Configuration - ENABLED
ENSEMBLE_CONFIG = {
    "use_ensemble": True,  # ✅ ENABLED - Use weighted ensemble voting
    "voting_strategy": "weighted",
    "temperature": 1.5,
    "confidence_threshold": 0.75,
    "agreement_penalty": True,
    "calibration_enabled": True
}


# Pipeline Configuration
PIPELINE_CONFIG = {
    "max_text_length": 10000,
    "batch_size": 1,
    "num_workers": 0,
    "fact_check_enabled": True,
    "domain_check_enabled": True,
    "content_analysis_enabled": True,
    "pattern_detection_enabled": True,
    "external_apis_enabled": True  # NEW for v4.0
}


# External API Configuration
EXTERNAL_API_CONFIG = {
    "timeout": 5,  # seconds
    "retry_attempts": 2,
    "fallback_mode_enabled": True,
    "cache_enabled": False
}


# Accuracy Targets
ACCURACY_TARGETS = {
    "overall": 0.92,  # 92% target with external APIs
    "high_confidence": 0.98,  # 98% for high confidence cases
    "medium_confidence": 0.87,  # 82-88% for medium confidence
    "low_confidence": 0.75  # 70-80% for low confidence
}


# Language Support
SUPPORTED_LANGUAGES = ["en", "ur", "ps", "hi"]


# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S"
}
