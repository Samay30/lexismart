"""backend/routers/health.py — Health check endpoint."""

from fastapi import APIRouter
from schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Simple health check — returns status and version."""
    return HealthResponse(status="ok", version="2.0.0")