"""
Analysis History Router
Track and export user's analysis history
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import json
import io

router = APIRouter()

# In-memory storage (replace with database in production)
history_store: List[Dict] = []


class AnalysisRecord(BaseModel):
    id: str
    timestamp: str
    text_preview: str
    result: str
    confidence: float
    analysis_type: str


@router.get("/history")
async def get_history():
    """Get user's analysis history (last 50)"""
    return {"history": history_store[-50:], "total": len(history_store)}


@router.post("/history")
async def save_analysis(record: AnalysisRecord):
    """Save analysis to history"""
    history_store.append(record.dict())
    return {"success": True, "total": len(history_store)}


@router.get("/export/json")
async def export_json():
    """Export history as JSON"""
    json_str = json.dumps(history_store, indent=2)
    return StreamingResponse(
        io.BytesIO(json_str.encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=neurolex_history.json"}
    )


@router.get("/export/csv")
async def export_csv():
    """Export history as CSV"""
    if not history_store:
        raise HTTPException(400, "No history to export")
    
    csv_lines = ["ID,Timestamp,Preview,Result,Confidence,Type"]
    for record in history_store:
        preview = record['text_preview'][:50].replace('"', '""')
        csv_lines.append(
            f"{record['id']},{record['timestamp']},"
            f'"{preview}",'
            f"{record['result']},{record['confidence']},{record['analysis_type']}"
        )
    
    csv_str = "\n".join(csv_lines)
    return StreamingResponse(
        io.BytesIO(csv_str.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=neurolex_history.csv"}
    )


@router.delete("/history")
async def clear_history():
    """Clear all history"""
    history_store.clear()
    return {"success": True, "message": "History cleared"}


@router.get("/stats")
async def get_stats():
    """Get analysis statistics"""
    if not history_store:
        return {"total": 0, "fake": 0, "real": 0}
    
    fake_count = sum(1 for r in history_store if r['result'].upper() == 'FAKE')
    real_count = sum(1 for r in history_store if r['result'].upper() == 'REAL')
    
    return {
        "total": len(history_store),
        "fake": fake_count,
        "real": real_count,
        "fake_percentage": (fake_count / len(history_store) * 100) if history_store else 0
    }
