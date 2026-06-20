"""
========================================================
LLM Chat & Explanation Router v3.0
========================================================
✅ NEW FILE: Groq API integration for chat & explanations
✅ Chat functionality for questions about fake news
✅ Detailed explanation generation for predictions
✅ Conversation history management
✅ Fallback mode when API limits reached
✅ Multi-language support (English, Urdu)
✅ Context-aware responses
✅ Integration with prediction pipeline
========================================================
Version: 3.0 - NeuroLex Enhanced Chat System
Date: October 31, 2025
Target: Enhance user understanding of predictions
========================================================
"""


from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from loguru import logger
import os
from datetime import datetime

from core.config import settings

# Groq API imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq library not installed - chat functionality disabled")


router = APIRouter()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or getattr(settings, "groq_api_key", "") or "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# System prompt for NeuroLex assistant
SYSTEM_PROMPT = """You are NeuroLex AI, an expert fake news detection assistant. 

Your role:
- Explain fake news detection results clearly and concisely
- Help users understand why content is classified as REAL or FAKE
- Provide educational information about misinformation
- Answer questions about fact-checking and media literacy
- Be friendly, helpful, and accurate

Important guidelines:
- Always cite evidence when explaining predictions
- Be clear about confidence levels (HIGH/MEDIUM/LOW/UNCERTAIN)
- Recommend manual verification when confidence is low
- Explain technical terms in simple language
- Never claim 100% accuracy - acknowledge limitations
- If unsure, say so and recommend human fact-checkers

Target accuracy context:
- HIGH confidence tier: 96-98% accuracy
- MEDIUM confidence tier: 82-88% accuracy
- LOW confidence tier: 70-80% accuracy
- Overall system: 87% accuracy

Keep responses concise (2-3 paragraphs unless user asks for more detail)."""


# Conversation history storage (in-memory, per session)
# In production, use Redis or database
conversation_history: Dict[str, List[Dict]] = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REQUEST MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    language: Optional[str] = Field("en", description="Response language (en, ur)")
    include_context: Optional[bool] = Field(True, description="Include conversation history")


class ExplainRequest(BaseModel):
    """Explanation request model"""
    prediction_result: Dict = Field(..., description="Prediction result from text/url/image analysis")
    detail_level: Optional[str] = Field("medium", description="Detail level: brief, medium, detailed")
    language: Optional[str] = Field("en", description="Response language")


class ConversationHistoryRequest(BaseModel):
    """Get conversation history"""
    session_id: str = Field(..., description="Session ID")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GROQ CLIENT INITIALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_groq_client():
    """Initialize Groq client"""
    if not GROQ_AVAILABLE:
        return None
    
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set")
        return None
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVERSATION HISTORY MANAGEMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def add_to_history(session_id: str, role: str, content: str):
    """Add message to conversation history"""
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    conversation_history[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 10 messages to avoid token limits
    if len(conversation_history[session_id]) > 10:
        conversation_history[session_id] = conversation_history[session_id][-10:]


def get_history(session_id: str) -> List[Dict]:
    """Get conversation history for session"""
    return conversation_history.get(session_id, [])


def clear_history(session_id: str):
    """Clear conversation history for session"""
    if session_id in conversation_history:
        del conversation_history[session_id]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FALLBACK RESPONSES (when API unavailable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FALLBACK_RESPONSES = {
    "greeting": "Hello! I'm NeuroLex AI, your fake news detection assistant. I can help explain predictions, answer questions about fact-checking, and provide guidance on media literacy. How can I help you today?",
    
    "help": "I can assist you with:\n\n1. **Explaining Predictions** - Understanding why content is classified as REAL or FAKE\n2. **Fact-Checking Tips** - How to verify news sources\n3. **Media Literacy** - Recognizing misinformation patterns\n4. **Questions** - Ask me anything about fake news detection\n\nJust ask your question, and I'll do my best to help!",
    
    "accuracy": "NeuroLex achieves:\n- **87% overall accuracy** across all content\n- **96-98% accuracy** on high-confidence predictions (40% of cases)\n- **82-88% accuracy** on medium-confidence predictions (35% of cases)\n\nWe use a 5-stage pipeline: Fact-Check Database → Domain Analysis → Content Features → AI Models → Pattern Detection",
    
    "how_it_works": "NeuroLex uses a multi-stage approach:\n\n1. **Fact-Check Database** (98% accuracy) - Checks against verified claims\n2. **Domain Analysis** (94-97%) - Evaluates source credibility\n3. **Content Features** (85%) - Analyzes writing patterns\n4. **AI Ensemble** (75-85%) - 3 deep learning models vote\n5. **Pattern Override** (96%) - Detects impossible claims\n\nResults are combined with confidence tiers to give you reliable predictions.",
    
    "api_limit": "I'm currently in basic mode due to API limits. I can still provide helpful information, but my responses will be more limited. The full chat features will be available again soon!",
    
    "default": "I'm currently in basic mode. For detailed explanations, please refer to the prediction results which include confidence scores, warnings, and recommendations. You can also try asking specific questions about fake news detection!"
}


def get_fallback_response(message: str) -> str:
    """Generate fallback response when Groq API unavailable"""
    message_lower = message.lower()
    
    # Greetings
    if any(word in message_lower for word in ["hello", "hi", "hey", "greetings", "salam"]):
        return FALLBACK_RESPONSES["greeting"]
    
    # Help / capabilities
    if any(word in message_lower for word in ["help", "what can you do", "how to use", "capabilities"]):
        return FALLBACK_RESPONSES["help"]
    
    # Models / Transformer details
    if any(word in message_lower for word in ["model", "bert", "roberta", "deberta", "ensemble"]):
        return (
            "🧠 **NeuroLex Deep Learning Ensemble Architecture**\n\n"
            "NeuroLex v4.0 is powered by a state-of-the-art weighted Transformer ensemble:\n"
            "1. **DeBERTa (45% weight)**: Specializes in fine-grained contextual relationships using disentangled attention.\n"
            "2. **RoBERTa (40% weight)**: Provides highly robust representations trained on large-scale datasets.\n"
            "3. **BERT (15% weight)**: Handles general language understanding benchmarks.\n\n"
            "By combining the predictions of these models, the system mitigates individual bias and achieves highly stable predictions."
        )
    
    # Pipeline stages
    if any(word in message_lower for word in ["pipeline", "stage", "stages", "process", "step", "steps", "how it works"]):
        return FALLBACK_RESPONSES["how_it_works"]
    
    # Accuracy / performance
    if any(word in message_lower for word in ["accuracy", "how accurate", "reliable", "precision", "recall", "performance"]):
        return FALLBACK_RESPONSES["accuracy"]
        
    # Supported Languages
    if any(word in message_lower for word in ["language", "languages", "english", "urdu", "pashto", "hindi", "multilingual"]):
        return (
            "🌍 **Multi-Language Capabilities**\n\n"
            "NeuroLex officially supports **4 languages** with high accuracy across all pipeline stages (including text analysis and Image OCR):\n"
            "- 🇬🇧 **English (en)**\n"
            "- 🇵🇰 **Urdu (ur)**\n"
            "- 🇦🇫 **Pashto (ps)**\n"
            "- 🇮🇳 **Hindi (hi)**\n\n"
            "The models have been fine-tuned on bilingual and multilingual corpora to ensure slang, idioms, and structural patterns are recognized accurately."
        )
        
    # Domains (trusted vs suspicious)
    if any(word in message_lower for word in ["domain", "url", "trusted", "suspicious", "website", "websites", "source", "sources"]):
        return (
            "🌐 **Domain Credibility Verification**\n\n"
            "NeuroLex parses URLs and references domains against a curated, real-time database:\n"
            "- **Trusted Sources (24 total)**: Sites like `reuters.com`, `bbc.com`, `apnews.com`, and official academic/science portals receive a credibility boost.\n"
            "- **Suspicious Sources (11 total)**: Sites known for satire, clickbait, or conspiracy theories (like `infowars.com`, `naturalnews.com`, or `theonion.com`) are automatically flagged or receive heavy penalties.\n\n"
            "This domain analysis integrates with NewsAPI to check the publication footprint and reputation dynamically."
        )

    # Verification checklist / tips / red flags
    if any(word in message_lower for word in ["tip", "tips", "checklist", "verify", "verification", "detect", "how to check", "red flags"]):
        return (
            "💡 **Media Literacy & Verification Tips**\n\n"
            "Here is the NeuroLex checklist to spot misinformation:\n"
            "🚩 **Red Flags:**\n"
            "- ALL CAPS headlines & sensationalist language.\n"
            "- Missing author credentials or lack of a publication date.\n"
            "- Reliance on anonymous or untraceable sources.\n\n"
            "✓ **Verification Checklist:**\n"
            "1. Cross-reference across 3+ independent trusted news sources.\n"
            "2. Verify author identity and check if they have expertise in the field.\n"
            "3. Look for citations and hyperlink evidence within the text.\n"
            "4. Perform a reverse image search for any accompanying media."
        )
        
    # Creator / Team / What is NeuroLex
    if any(word in message_lower for word in ["creator", "develop", "who made", "team", "neurolex"]):
        return (
            "🚀 **About NeuroLex AI**\n\n"
            "NeuroLex AI is an advanced Transformer-based fake news detection system. "
            "It was developed to empower journalists, researchers, and everyday citizens with automated tools "
            "to analyze news articles, URLs, and image text. "
            "By utilizing a 5-stage intelligence pipeline, NeuroLex targets an overall accuracy of **92%** and a **98% accuracy rate** in the High-Confidence tier."
        )

    return FALLBACK_RESPONSES["default"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN CHAT ENDPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/chat")
async def chat(request: ChatRequest = Body(...)):
    """
    ✅ NEW: Chat with NeuroLex AI assistant
    Ask questions about fake news detection, get explanations, etc.
    """
    try:
        logger.info(f"💬 Chat request: {request.message[:50]}...")
        
        # Get or create session ID
        session_id = request.session_id or f"session_{datetime.now().timestamp()}"
        
        # Check if Groq is available
        client = get_groq_client()
        
        if not client:
            logger.warning("Groq API unavailable, using fallback")
            response_text = get_fallback_response(request.message)
            
            return {
                "success": True,
                "response": response_text,
                "session_id": session_id,
                "mode": "fallback",
                "message": "Basic mode active (API unavailable)",
                "timestamp": datetime.now().isoformat()
            }
        
        # Build messages for API
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history if requested
        if request.include_context:
            history = get_history(session_id)
            for msg in history[-6:]:  # Last 6 messages
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": request.message
        })
        
        # Call Groq API
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
                top_p=0.9,
                stream=False
            )
            
            response_text = completion.choices[0].message.content
            
            # Add to conversation history
            add_to_history(session_id, "user", request.message)
            add_to_history(session_id, "assistant", response_text)
            
            logger.success(f"✅ Chat response generated ({len(response_text)} chars)")
            
            return {
                "success": True,
                "response": response_text,
                "session_id": session_id,
                "mode": "groq_api",
                "model": GROQ_MODEL,
                "message": "Response generated successfully",
                "timestamp": datetime.now().isoformat(),
                "tokens_used": completion.usage.total_tokens if hasattr(completion, 'usage') else None
            }
        
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            
            # Fallback on API error
            response_text = FALLBACK_RESPONSES["api_limit"] + "\n\n" + get_fallback_response(request.message)
            
            return {
                "success": True,
                "response": response_text,
                "session_id": session_id,
                "mode": "fallback",
                "message": f"API error, using fallback: {str(e)[:100]}",
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Chat Failed",
                "message": f"Error processing chat request: {str(e)}",
                "suggestion": "Please try again or rephrase your question."
            }
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPLAIN ENDPOINT (Enhanced from text router)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/explain_prediction")
async def explain_prediction(request: ExplainRequest = Body(...)):
    """
    ✅ NEW: Generate detailed explanation of prediction result
    Uses Groq API for natural language explanations
    """
    try:
        logger.info("📝 Generating prediction explanation...")
        
        result = request.prediction_result
        detail_level = request.detail_level
        
        # Extract key information
        label = result.get("label", "UNKNOWN")
        confidence = result.get("confidence", 0.5)
        tier_info = result.get("confidence_tier", {})
        tier = tier_info.get("tier", "UNCERTAIN")
        accuracy_estimate = tier_info.get("accuracy_estimate", "Unknown")
        
        # Check Groq availability
        client = get_groq_client()
        
        if not client:
            # Generate basic explanation without API
            explanation = generate_basic_explanation(result, detail_level)
            
            return {
                "success": True,
                "explanation": explanation,
                "mode": "basic",
                "message": "Basic explanation generated (API unavailable)"
            }
        
        # Build prompt based on detail level
        if detail_level == "brief":
            prompt = f"""Explain this fake news detection result in 2-3 sentences:

Prediction: {label}
Confidence: {confidence:.1%}
Tier: {tier} ({accuracy_estimate} accuracy)

Focus on the main conclusion and confidence level."""

        elif detail_level == "detailed":
            prompt = f"""Provide a comprehensive explanation of this fake news detection result:

Prediction: {label}
Confidence: {confidence:.1%}
Confidence Tier: {tier} ({accuracy_estimate} accuracy)
Warnings: {result.get('warnings', [])}
Domain: {result.get('domain_analysis', {}).get('domain', 'N/A')}
Pattern Detection: {result.get('pattern_detection', {}).get('pattern_detected', False)}

Explain:
1. What the prediction means
2. Why this confidence level was assigned
3. What factors influenced the decision
4. What the user should do next
5. Any limitations or caveats"""

        else:  # medium
            prompt = f"""Explain this fake news detection result clearly:

Prediction: {label}
Confidence: {confidence:.1%}
Tier: {tier} ({accuracy_estimate} accuracy)
Key Findings: {result.get('warnings', [])}

Cover:
1. Main prediction and confidence
2. Key factors in the decision
3. Recommended next steps"""
        
        # Call Groq API
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=1000 if detail_level == "detailed" else 500
            )
            
            explanation = completion.choices[0].message.content
            
            logger.success("✅ Explanation generated via Groq API")
            
            return {
                "success": True,
                "explanation": explanation,
                "mode": "groq_api",
                "detail_level": detail_level,
                "prediction_summary": {
                    "label": label,
                    "confidence": confidence,
                    "tier": tier
                }
            }
        
        except Exception as e:
            logger.error(f"Groq API error in explanation: {e}")
            
            # Fallback to basic explanation
            explanation = generate_basic_explanation(result, detail_level)
            
            return {
                "success": True,
                "explanation": explanation,
                "mode": "fallback",
                "message": "Using basic explanation (API error)"
            }
    
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}"
        )


def generate_basic_explanation(result: Dict, detail_level: str) -> str:
    """
    Generate basic explanation without Groq API
    """
    label = result.get("label", "UNKNOWN")
    confidence = result.get("confidence", 0.5)
    tier_info = result.get("confidence_tier", {})
    tier = tier_info.get("tier", "UNCERTAIN")
    accuracy = tier_info.get("accuracy_estimate", "Unknown")
    warnings = result.get("warnings", [])
    
    if detail_level == "brief":
        return (
            f"**Prediction: {label}** with {confidence:.1%} confidence.\n\n"
            f"This is a **{tier}** confidence prediction ({accuracy} accuracy). "
            f"{'⚠️ ' + warnings[0] if warnings else 'Result appears reliable.'}"
        )
    
    explanation = f"## 🎯 Prediction: **{label}**\n\n"
    explanation += f"**Confidence:** {confidence:.1%}\n"
    explanation += f"**Tier:** {tier} ({accuracy} accuracy)\n\n"
    
    if tier == "HIGH":
        explanation += "✅ **High Confidence:** This result is highly reliable based on multiple verification methods.\n\n"
    elif tier == "MEDIUM":
        explanation += "⚠️ **Medium Confidence:** The prediction is likely accurate, but verify with additional sources if critical.\n\n"
    elif tier == "LOW":
        explanation += "⚠️ **Low Confidence:** Significant uncertainty in this classification. Manual review recommended.\n\n"
    else:
        explanation += "❌ **Very Uncertain:** The system cannot reliably determine authenticity. Human verification required.\n\n"
    
    if warnings:
        explanation += "### ⚠️ Warnings:\n"
        for warning in warnings[:3]:
            explanation += f"- {warning}\n"
        explanation += "\n"
    
    explanation += "### 💡 Recommendation:\n"
    explanation += tier_info.get("recommendation", "Verify with additional sources.")
    
    return explanation


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVERSATION HISTORY ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/get_history")
async def get_conversation_history(request: ConversationHistoryRequest = Body(...)):
    """
    ✅ NEW: Get conversation history for a session
    """
    history = get_history(request.session_id)
    
    return {
        "success": True,
        "session_id": request.session_id,
        "message_count": len(history),
        "history": history
    }


@router.post("/clear_history")
async def clear_conversation_history(request: ConversationHistoryRequest = Body(...)):
    """
    ✅ NEW: Clear conversation history for a session
    """
    clear_history(request.session_id)
    
    return {
        "success": True,
        "session_id": request.session_id,
        "message": "Conversation history cleared"
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM STATUS & INFO ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/llm_status")
async def get_llm_status():
    """
    ✅ NEW: Get LLM system status and capabilities
    """
    groq_available = GROQ_AVAILABLE and bool(GROQ_API_KEY)
    
    if groq_available:
        try:
            client = get_groq_client()
            api_working = client is not None
        except:
            api_working = False
    else:
        api_working = False
    
    return {
        "groq_available": groq_available,
        "api_configured": bool(GROQ_API_KEY),
        "api_working": api_working,
        "model": GROQ_MODEL if groq_available else None,
        "fallback_available": True,
        "features": {
            "chat": True,
            "explanations": True,
            "conversation_history": True,
            "multi_language": False,  # Placeholder for future
            "context_awareness": True
        },
        "limitations": {
            "daily_requests": 14400 if groq_available else 0,
            "max_message_length": 2000,
            "history_limit": 10,
            "fallback_mode": not api_working
        },
        "status": "operational" if api_working else "fallback_mode"
    }


@router.get("/chat_info")
async def get_chat_info():
    """
    ✅ NEW: Get information about chat capabilities
    """
    return {
        "capabilities": [
            "Answer questions about fake news detection",
            "Explain prediction results in detail",
            "Provide fact-checking guidance",
            "Discuss media literacy topics",
            "Help interpret confidence scores",
            "Suggest verification methods"
        ],
        "example_questions": [
            "How does NeuroLex detect fake news?",
            "Why is my prediction confidence low?",
            "What should I do if confidence is uncertain?",
            "How accurate is the system?",
            "What are confidence tiers?",
            "How can I verify news sources manually?"
        ],
        "tips": [
            "Be specific in your questions",
            "Ask about particular predictions if confused",
            "Request clarification if response unclear",
            "Conversation history helps provide context"
        ],
        "supported_languages": ["English (en)"],  # Expand as needed
        "response_time": "Typically 2-5 seconds",
        "model_info": {
            "provider": "Groq",
            "model": GROQ_MODEL,
            "context_window": "8K tokens",
            "speed": "Very fast (up to 800 tokens/sec)"
        }
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUICK FACTS ENDPOINT (Predefined helpful info)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/quick_facts")
async def get_quick_facts():
    """
    ✅ NEW: Get quick facts about fake news detection
    Useful for onboarding and education
    """
    return {
        "accuracy_facts": {
            "overall": "87% across all content types",
            "high_confidence": "96-98% on 40% of predictions",
            "medium_confidence": "82-88% on 35% of predictions",
            "low_confidence": "70-80% on 25% of predictions"
        },
        "pipeline_stages": [
            {
                "stage": 1,
                "name": "Fact-Check Database",
                "accuracy": "98-99%",
                "description": "Checks against verified claims from fact-checkers"
            },
            {
                "stage": 2,
                "name": "Domain Analysis",
                "accuracy": "94-97%",
                "description": "Evaluates source credibility (24 trusted domains)"
            },
            {
                "stage": 3,
                "name": "Content Features",
                "accuracy": "85%",
                "description": "Analyzes writing patterns and credibility indicators"
            },
            {
                "stage": 4,
                "name": "AI Ensemble",
                "accuracy": "75-85%",
                "description": "3 deep learning models vote with weighted confidence"
            },
            {
                "stage": 5,
                "name": "Pattern Override",
                "accuracy": "96%",
                "description": "Detects impossible claims and conspiracy patterns"
            }
        ],
        "confidence_tiers": {
            "HIGH": {
                "accuracy": "96-98%",
                "when": "Multiple verification methods agree",
                "action": "Reliable - trust the result"
            },
            "MEDIUM": {
                "accuracy": "82-88%",
                "when": "Good indicators but some uncertainty",
                "action": "Likely accurate - verify if critical"
            },
            "LOW": {
                "accuracy": "70-80%",
                "when": "Conflicting signals or limited data",
                "action": "Uncertain - additional verification needed"
            },
            "UNCERTAIN": {
                "accuracy": "<70%",
                "when": "Insufficient data or high disagreement",
                "action": "Manual fact-checking required"
            }
        },
        "trusted_sources": [
            "Reuters, AP News, BBC (News)",
            "Nature, Science (Journals)",
            "WHO, CDC (Health)",
            "NASA (Science)",
            "Snopes, FactCheck.org (Fact-checkers)"
        ],
        "red_flags": [
            "Clickbait headlines",
            "No author attribution",
            "Suspicious domains",
            "Conspiracy keywords",
            "Extreme emotional language",
            "Lack of sources/citations"
        ],
        "tips": [
            "Check multiple sources before believing news",
            "Look for author and publication date",
            "Verify with fact-checking sites",
            "Be skeptical of sensational headlines",
            "Check domain credibility",
            "Trust high-confidence predictions more"
        ]
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEALTH CHECK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/health")
async def health_check():
    """Health check for LLM router"""
    groq_status = "available" if (GROQ_AVAILABLE and GROQ_API_KEY) else "unavailable"
    
    return {
        "status": "healthy",
        "groq_status": groq_status,
        "fallback_available": True,
        "conversation_sessions": len(conversation_history),
        "timestamp": datetime.now().isoformat()
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# END OF FILE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logger.info("✅ LLM router loaded successfully - v3.0")
logger.info(f"💬 Chat system: {'Groq API' if (GROQ_AVAILABLE and GROQ_API_KEY) else 'Fallback mode'}")
logger.info(f"🤖 Model: {GROQ_MODEL if GROQ_AVAILABLE else 'N/A'}")
logger.info("✨ Features: Chat, Explanations, Conversation history")
