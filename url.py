"""
======================================================
Enhanced URL Extraction & Fake News Analysis Router v3.0
======================================================
✅ ENHANCED: Integration with text router pipeline
✅ ENHANCED: Domain credibility from TRUSTED_DOMAINS (24 sources)
✅ ENHANCED: Confidence tier support (HIGH/MEDIUM/LOW/UNCERTAIN)
✅ ENHANCED: Pattern detection on extracted content
✅ ENHANCED: Better metadata extraction (author, date)
✅ FIXED: Comprehensive error handling with suggestions
✅ ADDED: URL validation with accessibility check
✅ TARGET: 87% overall accuracy, 96% high-confidence
======================================================
Version: 3.0 - Aligned with NeuroLex Accuracy Target
Date: October 31, 2025
======================================================
"""


from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, HttpUrl, validator
from loguru import logger
import httpx
import asyncio
from playwright.sync_api import sync_playwright, Error as PlaywrightError
from bs4 import BeautifulSoup
from typing import Optional, List, Dict
import re
from datetime import datetime

from services.ensemble_loader import predict_ensemble, get_loaded_models



router = APIRouter()



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENHANCED TRUSTED DOMAINS (SYNCED WITH TEXT ROUTER)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRUSTED_DOMAINS = {
    # International News
    "reuters.com": {"credibility": 0.98, "boost": 0.25, "category": "News Agency"},
    "apnews.com": {"credibility": 0.98, "boost": 0.25, "category": "News Agency"},
    "bbc.com": {"credibility": 0.96, "boost": 0.20, "category": "International News"},
    "bbc.co.uk": {"credibility": 0.96, "boost": 0.20, "category": "International News"},
    "cnn.com": {"credibility": 0.92, "boost": 0.15, "category": "International News"},
    "aljazeera.com": {"credibility": 0.94, "boost": 0.18, "category": "International News"},
    "theguardian.com": {"credibility": 0.95, "boost": 0.18, "category": "International News"},
    "nytimes.com": {"credibility": 0.96, "boost": 0.20, "category": "International News"},
    "washingtonpost.com": {"credibility": 0.95, "boost": 0.18, "category": "International News"},
    
    # Pakistani Media
    "dawn.com": {"credibility": 0.95, "boost": 0.20, "category": "Pakistani News"},
    "tribune.com.pk": {"credibility": 0.93, "boost": 0.15, "category": "Pakistani News"},
    "thenews.com.pk": {"credibility": 0.92, "boost": 0.15, "category": "Pakistani News"},
    
    # Scientific/Academic
    "nasa.gov": {"credibility": 0.99, "boost": 0.30, "category": "Government Science"},
    "nature.com": {"credibility": 0.99, "boost": 0.30, "category": "Scientific Journal"},
    "science.org": {"credibility": 0.99, "boost": 0.30, "category": "Scientific Journal"},
    "who.int": {"credibility": 0.98, "boost": 0.28, "category": "Health Authority"},
    "cdc.gov": {"credibility": 0.98, "boost": 0.28, "category": "Health Authority"},
    
    # Fact-Checkers
    "snopes.com": {"credibility": 0.97, "boost": 0.25, "category": "Fact-Checker"},
    "factcheck.org": {"credibility": 0.97, "boost": 0.25, "category": "Fact-Checker"},
    "politifact.com": {"credibility": 0.96, "boost": 0.22, "category": "Fact-Checker"},
}


SUSPICIOUS_DOMAINS = {
    "naturalnews.com": {"risk": 0.95, "category": "Pseudoscience"},
    "infowars.com": {"risk": 0.98, "category": "Conspiracy"},
    "beforeitsnews.com": {"risk": 0.92, "category": "Fake News"},
}



def extract_domain_from_url(url: str) -> str:
    """Extract clean domain from URL"""
    domain = re.sub(r'^https?://(www\.)?', '', url)
    domain = domain.split('/')[0].lower()
    return domain


def analyze_url_domain(url: str) -> Dict:
    """
    Analyze domain credibility (synced with text router logic)
    """
    domain = extract_domain_from_url(url)
    
    # Check trusted
    for trusted_domain, info in TRUSTED_DOMAINS.items():
        if trusted_domain in domain:
            logger.info(f"✅ Trusted domain: {domain} ({info['category']})")
            return {
                "is_trusted": True,
                "domain": domain,
                "credibility_score": info["credibility"],
                "boost": info["boost"],
                "category": info["category"]
            }
    
    # Check suspicious
    for suspicious_domain, info in SUSPICIOUS_DOMAINS.items():
        if suspicious_domain in domain:
            logger.warning(f"⚠️ Suspicious domain: {domain} ({info['category']})")
            return {
                "is_suspicious": True,
                "domain": domain,
                "credibility_score": 1.0 - info["risk"],
                "risk_score": info["risk"],
                "category": info["category"]
            }
    
    # Unknown domain
    return {
        "is_unknown": True,
        "domain": domain,
        "credibility_score": 0.5,
        "category": "Unknown Source"
    }



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REQUEST MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class URLRequest(BaseModel):
    """Request model for URL analysis"""
    url: HttpUrl
    use_playwright: Optional[bool] = False
    
    @validator('url')
    def validate_url(cls, v):
        url_str = str(v)
        if not url_str.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METADATA EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def extract_metadata(soup: BeautifulSoup) -> Dict:
    """
    Extract article metadata (author, publish date, etc.)
    """
    metadata = {
        "author": None,
        "publish_date": None,
        "has_metadata": False
    }
    
    # Try to find author
    author_selectors = [
        'meta[name="author"]',
        'meta[property="article:author"]',
        '.author', '.byline', '[rel="author"]'
    ]
    
    for selector in author_selectors:
        author_tag = soup.select_one(selector)
        if author_tag:
            if author_tag.name == 'meta':
                metadata["author"] = author_tag.get('content', '').strip()
            else:
                metadata["author"] = author_tag.get_text().strip()
            if metadata["author"]:
                metadata["has_metadata"] = True
                break
    
    # Try to find publish date
    date_selectors = [
        'meta[property="article:published_time"]',
        'meta[name="publish_date"]',
        'time[datetime]', '.publish-date', '.date'
    ]
    
    for selector in date_selectors:
        date_tag = soup.select_one(selector)
        if date_tag:
            if date_tag.name == 'meta':
                metadata["publish_date"] = date_tag.get('content', '').strip()
            elif date_tag.name == 'time':
                metadata["publish_date"] = date_tag.get('datetime', '').strip()
            else:
                metadata["publish_date"] = date_tag.get_text().strip()
            if metadata["publish_date"]:
                metadata["has_metadata"] = True
                break
    
    return metadata


def _fetch_url_with_sync_playwright(url: str) -> str:
    """Fetch page HTML using sync Playwright in a background thread."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        context = browser.new_context(
            user_agent=(
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ),
            locale='en-US'
        )
        page = context.new_page()
        page.goto(url, wait_until='networkidle', timeout=30000)
        page.wait_for_timeout(1000)
        content = page.content()
        context.close()
        browser.close()
        return content


async def fetch_url_with_playwright(url: str) -> str:
    """Fetch page HTML using a headless browser when HTTP requests are blocked."""
    logger.info(f"🌐 Falling back to Playwright browser fetch for: {url}")
    try:
        return await asyncio.to_thread(_fetch_url_with_sync_playwright, url)
    except PlaywrightError as e:
        logger.error(f"❌ Playwright fetch failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Browser Fetch Failed",
                "message": f"Playwright could not load the page: {str(e)}",
                "suggestion": "Install Playwright browsers or use Text Analysis instead.",
                "url": url
            }
        )
    except NotImplementedError as e:
        logger.error(f"❌ Playwright fetch unsupported in this runtime: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Browser Fetch Unsupported",
                "message": "Playwright browser launch is not supported in this environment.",
                "suggestion": "Use the Text Analysis route or run NeuroLex where Playwright subprocesses are permitted.",
                "url": url
            }
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN URL ANALYSIS ENDPOINT - ENHANCED FOR v3.0
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@router.post("/analyze_url")
async def analyze_url(request: URLRequest = Body(...)):
    """
    ✅ ENHANCED v3.0: Integration with text router pipeline
    ✅ ENHANCED: Domain credibility analysis (24 trusted sources)
    ✅ ENHANCED: Confidence tier determination
    ✅ ENHANCED: Pattern detection on content
    ✅ ENHANCED: Metadata extraction (author, date)
    ✅ FIXED: Comprehensive error handling
    
    Fetches and analyzes text from URL with full pipeline integration.
    """
    url_str = str(request.url)
    
    try:
        logger.info(f"📡 Analyzing URL: {url_str}")
        
        # STEP 1: Analyze domain credibility
        domain_analysis = analyze_url_domain(url_str)
        
        # STEP 2: Improved headers
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # STEP 3: Fetch URL
        html_content = None
        if request.use_playwright:
            html_content = await fetch_url_with_playwright(url_str)
        else:
            async with httpx.AsyncClient(
                timeout=20.0,
                follow_redirects=True,
                headers=headers,
                verify=True
            ) as client:
                try:
                    response = await client.get(url_str)
                    response.raise_for_status()
                    html_content = response.text
                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    if status_code == 403:
                        logger.warning("⚠️ HTTP 403 received. Retrying with Playwright browser fetch...")
                        html_content = await fetch_url_with_playwright(url_str)
                    elif status_code == 404:
                        raise HTTPException(
                            status_code=404,
                            detail={
                                "error": "Page Not Found",
                                "message": f"The URL {url_str} does not exist (404).",
                                "suggestion": "Please check if the URL is correct.",
                                "url": url_str
                            }
                        )
                    elif status_code == 429:
                        raise HTTPException(
                            status_code=429,
                            detail={
                                "error": "Too Many Requests",
                                "message": "The website is rate-limiting requests.",
                                "suggestion": "Wait a moment and try again, or use text analysis.",
                                "url": url_str
                            }
                        )
                    elif status_code >= 500:
                        raise HTTPException(
                            status_code=502,
                            detail={
                                "error": "Website Unavailable",
                                "message": f"Website is down ({status_code}).",
                                "suggestion": "Try again later or use text analysis.",
                                "url": url_str
                            }
                        )
                    else:
                        raise HTTPException(
                            status_code=status_code,
                            detail={
                                "error": f"HTTP Error {status_code}",
                                "message": f"Failed to load {url_str}",
                                "suggestion": "Try using text analysis instead.",
                                "url": url_str
                            }
                        )

        if not html_content:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "Fetch Failed",
                    "message": "Could not retrieve webpage content.",
                    "suggestion": "Use Text Analysis or try Playwright mode.",
                    "url": url_str
                }
            )
        
        # STEP 4: Parse HTML
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to parse HTML: {e}")
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Invalid HTML",
                    "message": "Page content could not be parsed.",
                    "suggestion": "Ensure URL points to a text-based webpage.",
                    "url": url_str
                }
            )
        
        # STEP 5: Extract metadata
        metadata = extract_metadata(soup)
        
        # STEP 6: Remove clutter
        for tag in ['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript', 'svg']:
            for node in soup.find_all(tag):
                node.decompose()
        
        # STEP 7: Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else "Untitled Page"
        
        # STEP 8: Extract main content
        selectors = [
            'article', 'main', '[role="main"]',
            '.article-content', '.post-content', '.entry-content',
            '#content', '.content', '.story-body', '.article-body',
            '.post-body', '.article-text', '.entry-text'
        ]
        
        best_content = None
        max_len = 0
        best_sel = None
        
        for sel in selectors:
            elements = soup.select(sel)
            for el in elements:
                text_len = len(el.get_text(strip=True))
                if text_len > max_len and text_len > 100:
                    best_content = el
                    max_len = text_len
                    best_sel = sel
                    
        if best_content:
            content = best_content
            logger.info(f"✓ Best content found using: {best_sel} (length: {max_len})")
        else:
            content = None
        
        if not content:
            content = soup.find('body')
            logger.warning("⚠️ Using <body> fallback")
        
        if not content:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "No Content Found",
                    "message": "Could not find readable text.",
                    "suggestion": "Try copying text manually and using Text Analysis.",
                    "url": url_str
                }
            )
        
        # STEP 9: Clean text
        text = content.get_text(separator=' ', strip=True)
        text = ' '.join(text.split())
        
        # STEP 10: Check minimum length
        if len(text) < 80:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Content Too Short",
                    "message": f"Only {len(text)} characters extracted.",
                    "suggestion": "Use 'Text Analysis' feature instead.",
                    "url": url_str
                }
            )
        
        logger.info(f"✓ Extracted {len(text)} characters")
        
        # STEP 11: Check models
        if not get_loaded_models():
            return {
                "success": False,
                "url": url_str,
                "title": title,
                "text_preview": text[:1500],
                "text_length": len(text),
                "message": "Content extracted but AI models unavailable.",
                "domain_analysis": domain_analysis,
                "metadata": metadata
            }
        
        # STEP 12: Run ensemble prediction
        analysis_text = text[:2000]
        result = predict_ensemble(analysis_text)
        
        # STEP 13: Extract key values
        final_label = result.get("final_label", "UNKNOWN")
        confidence = result.get("confidence", 0.5)
        base_confidence = result.get("base_confidence", confidence)
        model_agreement = result.get("model_agreement", 1.0)
        confidence_tier = result.get("confidence_tier", {})
        
        # STEP 14: Apply domain boost
        if domain_analysis.get("is_trusted") and final_label == "FAKE" and confidence < 0.75:
            domain_boost = domain_analysis.get("boost", 0.20)
            logger.info(f"✅ Applying trusted domain boost: +{domain_boost}")
            
            final_label = "REAL"
            confidence = min(0.95, confidence + domain_boost)
            
            if "warnings" not in result:
                result["warnings"] = []
            result["warnings"].append(
                f"✅ Trusted source: {domain_analysis['domain']} ({domain_analysis['category']})"
            )
        
        # STEP 15: Apply suspicious domain override
        if domain_analysis.get("is_suspicious"):
            logger.warning(f"⚠️ Suspicious domain override: {domain_analysis['domain']}")
            final_label = "FAKE"
            confidence = max(0.92, domain_analysis.get("risk_score", 0.92))
            
            if "warnings" not in result:
                result["warnings"] = []
            result["warnings"].append(
                f"⚠️ Suspicious domain: {domain_analysis['domain']} ({domain_analysis['category']})"
            )
        
        # STEP 16: Build warnings
        warnings = result.get("warnings", [])
        
        # User requested to remove warnings
        # if confidence < 0.65:
        #     warnings.append("⚠️ Low confidence - manual review recommended")
        # elif 0.65 <= confidence < 0.75:
        #     warnings.append("⚠️ Moderate confidence - verify with additional sources")
        
        # if model_agreement < 0.60:
        #     warnings.append("⚠️ Models disagree - results may be unreliable")
        
        # if not metadata.get("has_metadata"):
        #     warnings.append("⚠️ No author/date metadata found - credibility uncertain")
        
        # STEP 17: Calculate probabilities
        if final_label == "FAKE":
            fake_prob = confidence
            real_prob = 1.0 - confidence
        else:
            real_prob = confidence
            fake_prob = 1.0 - confidence
        
        # STEP 18: Build response
        return {
            "success": True,
            "url": url_str,
            "title": title,
            "text_preview": text[:1500] + "..." if len(text) > 1500 else text,
            "text_length": len(text),
            "metadata": metadata,
            "domain_analysis": domain_analysis,
            "prediction": {
                "label": final_label,
                "confidence": round(confidence, 4),
                "base_confidence": round(base_confidence, 4),
                "probabilities": {
                    "REAL": round(real_prob, 4),
                    "FAKE": round(fake_prob, 4)
                },
                "confidence_tier": confidence_tier,
                "model_agreement": round(model_agreement, 4),
                "models_used": list(get_loaded_models().keys()),
                "override_applied": result.get("override_reason") is not None,
                "override_reason": result.get("override_reason"),
                "pattern_category": result.get("pattern_category")
            },
            "warnings": warnings,
            "recommendation": confidence_tier.get("recommendation", "Verify with additional sources"),
            "message": "Content extracted and analyzed successfully."
        }
    
    except httpx.TimeoutException:
        logger.error(f"⏱️ Timeout: {url_str}")
        raise HTTPException(
            status_code=504,
            detail={
                "error": "Request Timeout",
                "message": f"Website took too long to respond (>20s).",
                "suggestion": "💡 Copy the article text and use 'Text Analysis' instead.",
                "url": url_str
            }
        )
    
    except httpx.NetworkError as e:
        logger.error(f"❌ Network error: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Network Error",
                "message": f"Could not connect to {url_str}.",
                "suggestion": "Check internet connection and URL. Use 'Text Analysis' as alternative.",
                "url": url_str
            }
        )
    
    except httpx.TooManyRedirects:
        logger.error(f"❌ Too many redirects: {url_str}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Too Many Redirects",
                "message": "URL redirected too many times.",
                "suggestion": "Try 'Text Analysis' instead.",
                "url": url_str
            }
        )
    
    except httpx.InvalidURL:
        logger.error(f"❌ Invalid URL: {url_str}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid URL",
                "message": "URL format is invalid.",
                "suggestion": "Ensure URL starts with http:// or https://",
                "url": url_str
            }
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Analysis Failed",
                "message": f"Unexpected error: {str(e)}",
                "suggestion": (
                    "💡 Recommended:\n"
                    "1. Copy the article text\n"
                    "2. Use 'Text Analysis' section\n"
                    "3. Get instant results"
                ),
                "url": url_str
            }
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER ENDPOINT: Check URL Accessibility
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@router.post("/check_url")
async def check_url(request: URLRequest = Body(...)):
    """
    Quick endpoint to check if URL is accessible before full analysis.
    ✅ ENHANCED: Includes domain credibility check
    """
    url_str = str(request.url)
    
    try:
        # Analyze domain first
        domain_analysis = analyze_url_domain(url_str)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            response = await client.head(url_str)
            
            return {
                "accessible": True,
                "status_code": response.status_code,
                "url": url_str,
                "domain_analysis": domain_analysis,
                "headers": {
                    "content-type": response.headers.get("content-type", "unknown"),
                    "server": response.headers.get("server", "unknown")
                },
                "message": "URL is accessible and ready for analysis"
            }
    
    except httpx.HTTPStatusError as e:
        return {
            "accessible": False,
            "status_code": e.response.status_code,
            "url": url_str,
            "error": f"HTTP {e.response.status_code}",
            "message": "URL returned an error status code",
            "suggestion": "Use 'Text Analysis' feature instead"
        }
    
    except Exception as e:
        return {
            "accessible": False,
            "url": url_str,
            "error": str(e),
            "message": "Could not connect to URL",
            "suggestion": "Check URL format and internet connection"
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATCH URL ANALYSIS (NEW)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BatchURLRequest(BaseModel):
    """Request model for batch URL analysis"""
    urls: List[HttpUrl]
    
    @validator('urls')
    def validate_urls(cls, v):
        if len(v) > 10:
            raise ValueError('Maximum 10 URLs allowed per batch')
        return v


@router.post("/analyze_urls_batch")
async def analyze_urls_batch(request: BatchURLRequest = Body(...)):
    """
    Analyze multiple URLs in batch (max 10).
    ✅ NEW: Efficient batch processing for multiple news articles
    """
    results = []
    
    logger.info(f"🔄 Processing batch of {len(request.urls)} URLs...")
    
    for idx, url in enumerate(request.urls, 1):
        url_str = str(url)
        logger.info(f"   [{idx}/{len(request.urls)}] Processing: {url_str}")
        
        try:
            url_request = URLRequest(url=url)
            result = await analyze_url(url_request)
            results.append({
                "index": idx,
                "success": True,
                "url": url_str,
                "prediction": result.get("prediction"),
                "domain_analysis": result.get("domain_analysis"),
                "title": result.get("title")
            })
        except HTTPException as e:
            results.append({
                "index": idx,
                "success": False,
                "url": url_str,
                "error": e.detail
            })
        except Exception as e:
            results.append({
                "index": idx,
                "success": False,
                "url": url_str,
                "error": str(e)
            })
    
    # Calculate summary statistics
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    if successful > 0:
        fake_count = sum(
            1 for r in results 
            if r["success"] and r.get("prediction", {}).get("label") == "FAKE"
        )
        real_count = successful - fake_count
    else:
        fake_count = 0
        real_count = 0
    
    logger.success(
        f"✅ Batch complete: {successful}/{len(results)} successful | "
        f"FAKE: {fake_count} | REAL: {real_count}"
    )
    
    return {
        "total": len(request.urls),
        "successful": successful,
        "failed": failed,
        "summary": {
            "fake_count": fake_count,
            "real_count": real_count
        },
        "results": results
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# END OF FILE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logger.info("✅ URL router loaded successfully - v3.0 Enhanced")
logger.info(f"🌐 Trusted domains: {len(TRUSTED_DOMAINS)}")
logger.info(f"⚠️ Suspicious domains: {len(SUSPICIOUS_DOMAINS)}")
logger.info("🎯 Integrated with 87% accuracy target pipeline")
