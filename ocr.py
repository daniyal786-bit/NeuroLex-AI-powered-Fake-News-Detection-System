from io import BytesIO
from loguru import logger

# --- Safe Imports for OCR Libraries ---
try:
    import easyocr
except (ImportError, ModuleNotFoundError):
    easyocr = None

try:
    from PIL import Image
except (ImportError, ModuleNotFoundError):
    Image = None

try:
    import pytesseract
except (ImportError, ModuleNotFoundError):
    pytesseract = None

try:
    import cv2
except (ImportError, ModuleNotFoundError):
    cv2 = None

try:
    import numpy as np
except (ImportError, ModuleNotFoundError):
    np = None

# --- OCR Implementation ---

_reader = None  # Lazy-loaded EasyOCR reader

def _ensure_easyocr_reader(lang_list=['en']):
    """Initializes and returns the EasyOCR reader instance."""
    global _reader
    if _reader is not None:
        return True
    if not easyocr:
        logger.warning("EasyOCR not installed. OCR functionality will be limited.")
        return False
    try:
        logger.info("Initializing EasyOCR reader...")
        _reader = easyocr.Reader(lang_list, gpu=False)
        logger.info("EasyOCR reader initialized.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize EasyOCR reader: {e}")
        _reader = None
        return False

def image_to_text(content: bytes) -> str:
    """
    Performs OCR on image content using the best available library.
    Tries EasyOCR first, then falls back to Pytesseract.
    """
    # Attempt 1: EasyOCR (if available)
    if np and cv2 and _ensure_easyocr_reader():
        try:
            arr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("cv2 failed to decode image bytes")
            result = _reader.readtext(img, detail=0, paragraph=True)
            return "\n".join(result).strip()
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}. Falling back to Pytesseract.")

    # Attempt 2: Pytesseract (if available)
    if Image and pytesseract:
        try:
            img = Image.open(BytesIO(content)).convert("RGB")
            text = pytesseract.image_to_string(img)
            return (text or "").strip()
        except Exception as e:
            logger.error(f"Pytesseract OCR failed: {e}")

    logger.warning("All OCR methods failed or no OCR libraries are installed.")
    return ""
