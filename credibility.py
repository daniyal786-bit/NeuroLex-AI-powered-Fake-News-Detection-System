"""Credibility Router"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class CredibilityRequest(BaseModel):
    source: str


@router.post("/credibility")
async def credibility(request: CredibilityRequest):
    """Check source credibility (placeholder)."""
    return {
        "source": request.source,
        "credibility_score": 0.5,
        "message": "Credibility check not implemented",
        "status": "placeholder"
    }
