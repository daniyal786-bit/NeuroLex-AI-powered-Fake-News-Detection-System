"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEUROLEX AI v4.0 - EXTERNAL API INTEGRATION SERVICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Integrates Google Fact Check, NewsAPI, and TextRazor APIs
for enhanced fake news detection accuracy (87% → 92%+)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


import os
import asyncio
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from functools import lru_cache
import requests
from loguru import logger


# Try importing optional dependencies
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    logger.warning("NewsAPI not available - install with: pip install newsapi-python")


try:
    import textrazor
    TEXTRAZOR_AVAILABLE = True
except ImportError:
    TEXTRAZOR_AVAILABLE = False
    logger.warning("TextRazor not available - install with: pip install textrazor")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENVIRONMENT VARIABLES & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# Load API keys from environment
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TEXTRAZOR_API_KEY = os.getenv("TEXTRAZOR_API_KEY", "")


# Rate limits
GOOGLE_DAILY_LIMIT = int(os.getenv("GOOGLE_FACTCHECK_DAILY_LIMIT", "1000"))
NEWS_DAILY_LIMIT = int(os.getenv("NEWS_API_DAILY_LIMIT", "100"))
TEXTRAZOR_DAILY_LIMIT = int(os.getenv("TEXTRAZOR_DAILY_LIMIT", "500"))


# API call tracking
api_usage = {
    "google_factcheck": {"calls": 0, "errors": 0, "last_reset": datetime.now()},
    "newsapi": {"calls": 0, "errors": 0, "last_reset": datetime.now()},
    "textrazor": {"calls": 0, "errors": 0, "last_reset": datetime.now()}
}


# Initialize API clients
if NEWSAPI_AVAILABLE and NEWS_API_KEY:
    try:
        newsapi_client = NewsApiClient(api_key=NEWS_API_KEY)
        logger.info("✅ NewsAPI client initialized")
    except Exception as e:
        newsapi_client = None
        logger.warning(f"⚠️ NewsAPI initialization failed: {e}")
else:
    newsapi_client = None


if TEXTRAZOR_AVAILABLE and TEXTRAZOR_API_KEY:
    try:
        textrazor.api_key = TEXTRAZOR_API_KEY
        logger.info("✅ TextRazor API key configured")
    except Exception as e:
        logger.warning(f"⚠️ TextRazor configuration failed: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RATE LIMIT TRACKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def reset_daily_limits():
    """Reset API usage counters if 24 hours have passed."""
    now = datetime.now()
    for service in api_usage.values():
        if (now - service["last_reset"]).days >= 1:
            service["calls"] = 0
            service["errors"] = 0
            service["last_reset"] = now


def can_use_api(service: str, limit: int) -> bool:
    """Check if API can be used (within rate limit)."""
    reset_daily_limits()
    return api_usage[service]["calls"] < limit


def record_api_call(service: str, success: bool = True):
    """Record an API call for tracking."""
    api_usage[service]["calls"] += 1
    if not success:
        api_usage[service]["errors"] += 1


def get_api_stats() -> Dict[str, Any]:
    """Get current API usage statistics."""
    reset_daily_limits()
    return {
        "google_factcheck": {
            "calls_today": api_usage["google_factcheck"]["calls"],
            "limit": GOOGLE_DAILY_LIMIT,
            "remaining": GOOGLE_DAILY_LIMIT - api_usage["google_factcheck"]["calls"],
            "errors": api_usage["google_factcheck"]["errors"]
        },
        "newsapi": {
            "calls_today": api_usage["newsapi"]["calls"],
            "limit": NEWS_DAILY_LIMIT,
            "remaining": NEWS_DAILY_LIMIT - api_usage["newsapi"]["calls"],
            "errors": api_usage["newsapi"]["errors"]
        },
        "textrazor": {
            "calls_today": api_usage["textrazor"]["calls"],
            "limit": TEXTRAZOR_DAILY_LIMIT,
            "remaining": TEXTRAZOR_DAILY_LIMIT - api_usage["textrazor"]["calls"],
            "errors": api_usage["textrazor"]["errors"]
        }
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GOOGLE FACT CHECK API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def check_google_factcheck(claim: str) -> Optional[Dict[str, Any]]:
    """
    Query Google Fact Check Tools API for claim verification.
    
    Args:
        claim: Text claim to fact-check
        
    Returns:
        Dict with fact-check results or None if unavailable
    """
    if not GOOGLE_FACTCHECK_API_KEY:
        logger.debug("Google Fact Check API key not configured")
        return None
    
    if not can_use_api("google_factcheck", GOOGLE_DAILY_LIMIT):
        logger.warning("Google Fact Check API daily limit reached")
        return None
    
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": claim[:500],  # Limit query length
            "key": GOOGLE_FACTCHECK_API_KEY,
            "languageCode": "en"
        }
        
        response = requests.get(url, params=params, timeout=5)
        record_api_call("google_factcheck", success=response.status_code == 200)
        
        if response.status_code == 200:
            data = response.json()
            claims = data.get("claims", [])
            
            if claims:
                # Process first matching claim
                first_claim = claims[0]
                claim_review = first_claim.get("claimReview", [{}])[0]
                
                result = {
                    "found": True,
                    "claim_text": first_claim.get("text", ""),
                    "claimant": first_claim.get("claimant", "Unknown"),
                    "rating": claim_review.get("textualRating", ""),
                    "publisher": claim_review.get("publisher", {}).get("name", ""),
                    "url": claim_review.get("url", ""),
                    "review_date": claim_review.get("reviewDate", ""),
                    "language": claim_review.get("languageCode", "en")
                }
                
                logger.info(f"✅ Google Fact Check: Found claim with rating '{result['rating']}'")
                return result
            else:
                logger.debug("Google Fact Check: No matching claims found")
                return {"found": False}
        else:
            logger.warning(f"Google Fact Check API error: {response.status_code}")
            return None
            
    except Exception as e:
        record_api_call("google_factcheck", success=False)
        logger.error(f"Google Fact Check API exception: {str(e)}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NEWSAPI - DOMAIN VERIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def verify_news_domain(domain: str, keywords: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    Verify news domain credibility using NewsAPI.
    
    Args:
        domain: Domain to verify (e.g., 'bbc.com')
        keywords: Optional keywords to search for domain coverage
        
    Returns:
        Dict with domain verification results or None
    """
    if not newsapi_client:
        logger.debug("NewsAPI client not available")
        return None
    
    if not can_use_api("newsapi", NEWS_DAILY_LIMIT):
        logger.warning("NewsAPI daily limit reached")
        return None
    
    try:
        # Search for recent articles from this domain
        query = " OR ".join(keywords[:3]) if keywords else "news"
        
        response = newsapi_client.get_everything(
            q=query,
            domains=domain,
            language='en',
            sort_by='publishedAt',
            page_size=5
        )
        
        record_api_call("newsapi", success=True)
        
        articles = response.get('articles', [])
        total_results = response.get('totalResults', 0)
        
        result = {
            "domain": domain,
            "found": len(articles) > 0,
            "total_articles": total_results,
            "recent_articles": len(articles),
            "credibility_score": min(total_results / 100, 1.0),  # 0-1 score
            "articles": [
                {
                    "title": article.get("title", ""),
                    "published": article.get("publishedAt", ""),
                    "author": article.get("author", "")
                }
                for article in articles[:3]
            ]
        }
        
        logger.info(f"✅ NewsAPI: {domain} has {total_results} indexed articles")
        return result
        
    except Exception as e:
        record_api_call("newsapi", success=False)
        logger.error(f"NewsAPI exception: {str(e)}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEXTRAZOR - ENTITY EXTRACTION & NLP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def extract_entities_textrazor(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract entities and perform NLP analysis using TextRazor.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dict with entity extraction results or None
    """
    if not TEXTRAZOR_AVAILABLE or not TEXTRAZOR_API_KEY:
        logger.debug("TextRazor not available")
        return None
    
    if not can_use_api("textrazor", TEXTRAZOR_DAILY_LIMIT):
        logger.warning("TextRazor API daily limit reached")
        return None
    
    try:
        client = textrazor.TextRazor(extractors=["entities", "topics", "relations"])
        client.set_cleanup_mode("cleanHTML")
        
        # Limit text length
        text_sample = text[:10000]
        
        response = client.analyze(text_sample)
        record_api_call("textrazor", success=True)
        
        entities = []
        for entity in response.entities():
            entities.append({
                "text": entity.matched_text,
                "type": getattr(entity, 'freebase_types', ['Unknown'])[0] if hasattr(entity, 'freebase_types') else 'Unknown',
                "confidence": entity.confidence_score,
                "relevance": entity.relevance_score
            })
        
        topics = []
        if hasattr(response, 'topics'):
            for topic in response.topics():
                topics.append({
                    "label": topic.label,
                    "score": topic.score
                })
        
        result = {
            "entities": entities[:20],  # Top 20 entities
            "topics": topics[:10],      # Top 10 topics
            "entity_count": len(entities),
            "topic_count": len(topics)
        }
        
        logger.info(f"✅ TextRazor: Extracted {len(entities)} entities, {len(topics)} topics")
        return result
        
    except Exception as e:
        record_api_call("textrazor", success=False)
        logger.error(f"TextRazor exception: {str(e)}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMBINED VERIFICATION PIPELINE (FIXED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def enhanced_verification(
    text: str,
    source_url: Optional[str] = None,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Perform comprehensive verification using all available external APIs.
    
    Args:
        text: Text to verify
        source_url: Optional source URL to extract domain from
        language: Language code for fact-checking (default: "en")
        
    Returns:
        Combined verification results from all APIs
    """
    logger.info("🔍 Starting enhanced verification with external APIs")
    
    # Extract domain from source_url if provided
    domain = None
    if source_url:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(source_url)
            domain = parsed.netloc.replace('www.', '')
        except Exception:
            domain = None
    
    # Extract keywords from text (simple approach - first few important words)
    keywords = None
    if text:
        # Take first 5 words that are longer than 4 characters
        words = [w for w in text.split() if len(w) > 4]
        keywords = words[:5] if words else None
    
    # Run all API calls concurrently
    tasks = [
        check_google_factcheck(text),
        verify_news_domain(domain, keywords) if domain else None,
        extract_entities_textrazor(text)
    ]
    
    # Filter out None tasks
    tasks = [t for t in tasks if t is not None]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    fact_check = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else None
    domain_check = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
    entity_analysis = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None
    
    # Calculate a direction-aware signal based on API results.
    # IMPORTANT: routers/text.py will apply this signal relative to the current model_label.
    # We return a magnitude (confidence_adjustment) and a polarity (signal_support).
    #
    # signal_support values:
    #   "FAKE" -> evidence supports FAKE (increase FAKE confidence)
    #   "REAL" -> evidence supports REAL (decrease FAKE confidence)
    #   "UNCERTAIN" -> no reliable directional signal
    confidence_adjustment = 0.0
    signal_support = "UNCERTAIN"
    verification_flags: List[str] = []

    fake_evidence = 0.0
    real_evidence = 0.0

    if fact_check and fact_check.get("found"):
        rating = (fact_check.get("rating", "") or "").lower()
        if any(term in rating for term in ["false", "fake", "misleading", "unproven"]):
            fake_evidence += 0.15
            verification_flags.append("fact_check_negative")
        elif any(term in rating for term in ["true", "correct", "accurate", "verified"]):
            real_evidence += 0.10
            verification_flags.append("fact_check_positive")

    if domain_check and domain_check.get("found"):
        score = float(domain_check.get("credibility_score", 0) or 0)
        if score > 0.7:
            # Domain looks credible => evidence for REAL
            real_evidence += 0.05
            verification_flags.append("trusted_domain")
        elif score < 0.3:
            # Domain looks unreliable => evidence for FAKE
            fake_evidence += 0.10
            verification_flags.append("untrusted_domain")

    if entity_analysis:
        entity_count = int(entity_analysis.get("entity_count", 0) or 0)
        if entity_count > 15:  # Rich entity content suggests real news
            real_evidence += 0.03
            verification_flags.append("rich_entities")

    # Net polarity
    net = real_evidence - fake_evidence
    confidence_adjustment = abs(net)

    if confidence_adjustment > 0:
        signal_support = "REAL" if net > 0 else "FAKE"

    return {
        "fact_check": fact_check,
        "domain_verification": domain_check,
        "entity_analysis": entity_analysis,
        "confidence_adjustment": confidence_adjustment,
        "signal_support": signal_support,
        "verification_flags": verification_flags,
        "apis_used": {
            "google_factcheck": fact_check is not None,
            "newsapi": domain_check is not None,
            "textrazor": entity_analysis is not None
        },
        "language": language  # Include language in response
    }



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FALLBACK MODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def is_fallback_mode() -> bool:
    """Check if system is in fallback mode (all APIs exhausted)."""
    stats = get_api_stats()
    all_exhausted = all(
        stats[api]["remaining"] <= 0 
        for api in ["google_factcheck", "newsapi", "textrazor"]
    )
    return all_exhausted


def get_fallback_message() -> str:
    """Get user-friendly fallback message."""
    return (
        "⚠️ External API limits reached. "
        "Using local AI models only. "
        "Accuracy may be slightly reduced (87% vs 92%)."
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEALTH CHECK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def health_check() -> Dict[str, Any]:
    """
    Check health status of all external APIs.
    
    Returns:
        Health status of each API service
    """
    stats = get_api_stats()
    
    return {
        "status": "healthy" if not is_fallback_mode() else "degraded",
        "fallback_mode": is_fallback_mode(),
        "services": {
            "google_factcheck": {
                "available": bool(GOOGLE_FACTCHECK_API_KEY),
                "remaining_calls": stats["google_factcheck"]["remaining"],
                "errors": stats["google_factcheck"]["errors"]
            },
            "newsapi": {
                "available": newsapi_client is not None,
                "remaining_calls": stats["newsapi"]["remaining"],
                "errors": stats["newsapi"]["errors"]
            },
            "textrazor": {
                "available": TEXTRAZOR_AVAILABLE and bool(TEXTRAZOR_API_KEY),
                "remaining_calls": stats["textrazor"]["remaining"],
                "errors": stats["textrazor"]["errors"]
            }
        }
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_apis():
    """Test all configured external APIs."""
    print("\n" + "="*60)
    print("🧪 TESTING EXTERNAL APIs")
    print("="*60 + "\n")
    
    # Test Google Fact Check
    print("1️⃣ Testing Google Fact Check API...")
    fact_result = await check_google_factcheck("climate change is a hoax")
    print(f"   Result: {fact_result}\n")
    
    # Test NewsAPI
    print("2️⃣ Testing NewsAPI...")
    news_result = await verify_news_domain("bbc.com", ["technology"])
    print(f"   Result: {news_result}\n")
    
    # Test TextRazor
    print("3️⃣ Testing TextRazor...")
    entity_result = await extract_entities_textrazor(
        "Apple Inc. CEO Tim Cook announced new products in California."
    )
    print(f"   Result: {entity_result}\n")
    
    # Health check
    print("4️⃣ Health Check...")
    health = await health_check()
    print(f"   Status: {health}\n")
    
    print("="*60)
    print("✅ API TESTING COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_apis())
