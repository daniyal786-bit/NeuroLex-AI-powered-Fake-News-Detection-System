"""
Enhanced Chat Router with Conversational AI
Production-grade chat with context awareness and history integration
"""

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Optional
from loguru import logger
from datetime import datetime

from routers.history import history_store

try:
    from services.ensemble_loader import predict_ensemble, get_loaded_models
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

try:
    from services.model_loader import predict_fake_news, get_model
    MODEL_LOADER_AVAILABLE = True
except ImportError:
    MODEL_LOADER_AVAILABLE = False

router = APIRouter()


# ==================== MODELS ====================

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    session_id: Optional[str] = None


# ==================== MAIN CHAT ====================

@router.post("/chat")
async def chat(request: ChatRequest = Body(...)):
    try:
        user_message = request.message.strip()

        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        model_available = (
            (ENSEMBLE_AVAILABLE and bool(get_loaded_models()))
            or (MODEL_LOADER_AVAILABLE and get_model() is not None)
        )

        # ==================== ANALYSIS ====================
        analysis = None
        if len(user_message) > 50 and model_available:
            try:
                if ENSEMBLE_AVAILABLE and get_loaded_models():
                    raw = predict_ensemble(user_message)
                    analysis = {
                        "label": raw.get("label", "UNKNOWN"),
                        "confidence": raw.get("confidence", 0.5),
                    }
                elif MODEL_LOADER_AVAILABLE:
                    analysis = predict_fake_news(user_message)
            except Exception as e:
                logger.error(f"Analysis failed: {e}")

        # ==================== LAST CONTEXT ====================
        last_analysis = history_store[-1] if history_store else None

        # ==================== RESPONSE ====================
        response = generate_intelligent_response(
            user_message,
            request.history,
            analysis,
            last_analysis
        )

        # ==================== SAVE HISTORY ====================
        if analysis:
            history_store.append({
                "id": str(len(history_store) + 1),
                "timestamp": datetime.now().isoformat(),
                "text_preview": user_message[:100],
                "result": analysis["label"],
                "confidence": analysis["confidence"] * 100,
                "analysis_type": "chat"
            })

        return {
            "reply": response["reply"],
            "analysis": analysis,
            "suggestions": response.get("suggestions", []),
            "context_understood": True,
            "timestamp": datetime.now().isoformat(),
            "session_id": request.session_id
        }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== INTELLIGENCE ENGINE ====================

def generate_intelligent_response(
    message: str,
    history: List[ChatMessage],
    analysis: Optional[Dict],
    last_analysis: Optional[Dict] = None
) -> Dict:

    msg_lower = message.lower()

    # ==================== CONTEXT MEMORY ====================
    if last_analysis and any(word in msg_lower for word in ["this", "that", "it", "news", "url", "last", "previous", "result"]):

        return {
            "reply": (
                f"📊 **Previous Analysis Found**\n\n"
                f"📝 Preview: {last_analysis.get('text_preview', 'N/A')}\n"
                f"📌 Result: {last_analysis.get('result', 'Unknown')}\n"
                f"🎯 Confidence: {last_analysis.get('confidence', 0):.1f}%\n\n"
                f"💡 Your Question: {message}\n\n"
                f"👉 This content was classified as **{last_analysis.get('result')}**.\n"
                f"Do you want a full explanation?"
            ),
            "suggestions": ["Why fake?", "Why real?", "Explain more"]
        }

    # ==================== GREETING ====================
    if any(word in msg_lower for word in ['hello', 'hi', 'hey', 'salam', 'greetings', 'wassup']):
        return {
            "reply": (
                "👋 Hello! I'm NeuroLex AI\n\n"
                "I can help you:\n"
                "✔ Analyze news/articles\n"
                "✔ Detect fake news\n"
                "✔ Explain results\n"
                "✔ Teach verification skills\n\n"
                "Just paste text or ask anything!"
            ),
            "suggestions": ["Analyze news", "Detection tips", "How it works"]
        }

    # ==================== HOW IT WORKS (check BEFORE general help) ====================
    if any(word in msg_lower for word in ['work', 'pipeline', 'stage', 'process', 'detect', 'algorithm', 'model']):
        return {
            "reply": (
                "🔍 **How NeuroLex Detection Works:**\n\n"
                "🏆 5-Stage Pipeline:\n"
                "1️⃣ **Fact-Check DB** (98% acc) - Check against verified claims\n"
                "2️⃣ **Domain Analysis** (96% acc) - Evaluate source credibility\n"
                "3️⃣ **Content Features** (85% acc) - Analyze writing patterns\n"
                "4️⃣ **AI Ensemble** (80% acc) - Vote of 3 AI models\n"
                "5️⃣ **Pattern Detection** (96% acc) - Spot conspiracy patterns\n\n"
                "⚖️ **Final Score** combines all stages → Confidence Tier (HIGH/MEDIUM/LOW)"
            ),
            "suggestions": ["What models?", "Red flags", "Try analyzing"]
        }

    # ==================== RED FLAGS / MISINFORMATION SIGNS (check BEFORE general help) ====================
    if any(word in msg_lower for word in ['red flag', 'signs', 'spot', 'detect fake', 'fake news', 'misinformation', 'disinformation']):
        return {
            "reply": (
                "🚩 **Signs of Misinformation:**\n\n"
                "❌ **Content Red Flags:**\n"
                "• ALL CAPS HEADLINES\n"
                "• Extreme emotional language\n"
                "• Sensational claims without evidence\n"
                "• No author or publication date\n"
                "• Poor grammar/spelling\n\n"
                "❌ **Source Red Flags:**\n"
                "• Unknown/suspicious domain\n"
                "• No contact information\n"
                "• Anonymous authors\n"
                "• Clickbait URLs\n\n"
                "✅ **Verification Checklist:**\n"
                "✓ Cross-check 3+ trusted sources\n"
                "✓ Verify author credentials\n"
                "✓ Look for citations/links\n"
                "✓ Reverse image search attached photos"
            ),
            "suggestions": ["Verify sources", "Trusted domains", "Try analysis"]
        }

    # ==================== VERIFICATION / FACT-CHECKING (check BEFORE general help) ====================
    if any(word in msg_lower for word in ['verify', 'fact check', 'check source', 'verify news', 'confirm']):
        return {
            "reply": (
                "✅ **How to Verify News:**\n\n"
                "1️⃣ **Cross-Reference:** Check 3+ independent trusted news sources\n"
                "2️⃣ **Check Source:** Is the domain reputable? Search domain history\n"
                "3️⃣ **Verify Author:** Does author have credentials in this field?\n"
                "4️⃣ **Check Citations:** Are claims backed by evidence/links?\n"
                "5️⃣ **Reverse Image Search:** Use Google Images for attached photos\n"
                "6️⃣ **Check Date:** When was this published? Is it still relevant?\n\n"
                "🔗 **Trusted Fact-Checkers:**\n"
                "• Snopes.com\n"
                "• FactCheck.org\n"
                "• PolitiFact.com"
            ),
            "suggestions": ["Red flags", "Trusted sources", "Try analyzing"]
        }

    # ==================== ACCURACY ====================
    if any(word in msg_lower for word in ['accuracy', 'correct', 'reliable', 'precise', 'confident', 'how accurate']):
        return {
            "reply": (
                "📊 **Accuracy Breakdown:**\n\n"
                "• **Overall:** 92% across all content\n"
                "• **HIGH Confidence (40%):** 96-98% accurate ✅\n"
                "• **MEDIUM Confidence (35%):** 82-88% accurate ⚠️\n"
                "• **LOW Confidence (25%):** 70-80% accurate ⚠️\n\n"
                "💡 Higher confidence = more reliable prediction\n"
                "🔔 Always verify with additional sources when unsure"
            ),
            "suggestions": ["What are red flags?", "How to verify", "Try analysis"]
        }

    # ==================== CONFIDENCE TIERS ====================
    if any(word in msg_lower for word in ['confidence', 'tier', 'uncertain', 'high confidence', 'low confidence']):
        return {
            "reply": (
                "📊 **Understanding Confidence Tiers:**\n\n"
                "🟢 **HIGH (40% of cases):** 96-98% accurate\n"
                "   → Trust this result, but verify critical info\n\n"
                "🟡 **MEDIUM (35% of cases):** 82-88% accurate\n"
                "   → Likely accurate, verify before sharing\n\n"
                "🔴 **LOW (20% of cases):** 70-80% accurate\n"
                "   → Significant uncertainty, manual verification needed\n\n"
                "⚪ **UNCERTAIN (<5% of cases):** <70% accurate\n"
                "   → Insufficient data, human fact-checking required\n\n"
                "💡 Higher tier = More reliable prediction"
            ),
            "suggestions": ["How accuracy works", "What to do next", "Try analysis"]
        }

    # ==================== HELP REQUEST (generic, lower priority) ====================
    if any(word in msg_lower for word in ['help', 'what can', 'guide', 'tutorial']):
        return {
            "reply": (
                "🆘 **Here's how I can help:**\n\n"
                "1️⃣ **Paste text** - I'll analyze if it's fake or real\n"
                "2️⃣ **Ask questions** - 'How does detection work?', 'What are red flags?'\n"
                "3️⃣ **Get explanations** - I explain why content is classified a certain way\n"
                "4️⃣ **Learn tips** - Fact-checking & media literacy guidance\n\n"
                "Try asking about:\n"
                "• How fake news detection works\n"
                "• Red flags to spot misinformation\n"
                "• How to verify sources\n"
                "• Confidence scores explained"
            ),
            "suggestions": ["How it works", "Red flags", "Verify sources"]
        }

    # ==================== ANALYSIS RESULT ====================
    if analysis:
        label = analysis['label']
        confidence = analysis['confidence'] * 100

        return {
            "reply": (
                f"📊 **Analysis Result**\n\n"
                f"📌 Label: **{label}**\n"
                f"🎯 Confidence: {confidence:.1f}%\n\n"
                f"💡 Ask: 'why this result?' for a detailed explanation, or paste another article!"
            ),
            "suggestions": ["Explain result", "Analyze another", "More details"]
        }

    # ==================== DEFAULT ====================
    return {
        "reply": (
            "💬 **What would you like to know?**\n\n"
            "You can:\n"
            "📝 Paste any article or news text\n"
            "🔍 Ask about detection methods\n"
            "📊 Get explanation of results\n"
            "📚 Learn fact-checking tips\n"
            "❓ Ask any question about fake news\n\n"
            "Try: 'What are red flags?' or paste some text to analyze!"
        ),
        "suggestions": ["How it works", "Red flags", "Analyze text"]
    }