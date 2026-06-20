import httpx
from loguru import logger

from core.config import settings

async def translate_text(text: str, target_lang: str) -> dict:
    """
    Translate text to a target language using the LibreTranslate-compatible API.
    Languages are configured in settings.
    """
    if not (text or '').strip():
        return {"error": "No text provided", "translated_text": "", "detected_source": ""}

    raw = (target_lang or "en").strip().lower()
    # Normalize aliases
    target_lang = settings.translate_lang_aliases.get(raw, raw)

    if target_lang not in settings.translate_allowed_langs:
        allowed = ", ".join(sorted(list(settings.translate_allowed_langs)))
        return {"error": f"Unsupported language: {raw}. Allowed: {allowed}", "translated_text": "", "detected_source": ""}

    try:
        payload = {
            "q": text,
            "source": "auto",
            "target": target_lang,
            "format": "text"
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(settings.libre_url, json=payload)
            r.raise_for_status()
            res = r.json()
            translated = res.get("translatedText") or res.get("translation") or ""
            detected_source = res.get("detectedLanguage", {}).get("language", "auto")
            return {"translated_text": translated, "error": "", "detected_source": detected_source}
    except httpx.HTTPStatusError as e:
        logger.error(f"Translation API returned an error: {e.response.status_code} {e.response.text}")
        return {"error": f"Translation service failed with status {e.response.status_code}", "translated_text": "", "detected_source": ""}
    except Exception as e:
        logger.error(f"An unexpected error occurred during translation: {e}")
        return {"error": f"Translation failed: {str(e)}", "translated_text": "", "detected_source": ""}
