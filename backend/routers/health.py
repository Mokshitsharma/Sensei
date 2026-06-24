from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "timestamp": datetime.utcnow().isoformat()}
