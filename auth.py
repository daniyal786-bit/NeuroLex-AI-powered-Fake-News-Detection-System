from fastapi import APIRouter

router = APIRouter()

@router.get("/test")
async def test_auth():
    return {"ok": True}
