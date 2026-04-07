"""
routers/health.py — Health check endpoints.

GET /health          — liveness check (Render uses this)
GET /health/openai   — verifies OPENAI_API_KEY is set and reachable
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    status:  str
    version: str


class OpenAIHealthResponse(BaseModel):
    status:   str           # "ok" | "error"
    key_set:  bool          # is OPENAI_API_KEY present in env?
    message:  str


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check — returns 200 if the process is running."""
    return HealthResponse(status="ok", version="2.0.0")


@router.get("/health/openai", response_model=OpenAIHealthResponse)
async def health_openai() -> OpenAIHealthResponse:
    """
    Checks that OPENAI_API_KEY is configured and that OpenAI's API
    is reachable with it. Makes a minimal embedding call (1 token).

    Use this to diagnose "Service unavailable" errors before touching
    the main pipeline endpoints.

    curl https://lexismart.onrender.com/health/openai
    """
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY is not set")
        return OpenAIHealthResponse(
            status  = "error",
            key_set = False,
            message = "OPENAI_API_KEY is not set in environment variables. "
                      "Add it in Render → your service → Environment.",
        )

    # Make a minimal real API call to confirm the key works
    try:
        from openai_client import AsyncOpenAIClient
        client = AsyncOpenAIClient(
            api_key        = settings.openai_api_key,
            embed_model    = settings.openai_embed_model,
            retry_attempts = 1,   # don't retry on health check
            retry_delay_ms = 0,
        )
        await client.embed("ok")
        logger.info("OpenAI health check passed")
        return OpenAIHealthResponse(
            status  = "ok",
            key_set = True,
            message = "OPENAI_API_KEY is set and OpenAI API is reachable.",
        )
    except Exception as exc:
        real = str(exc)
        logger.error("OpenAI health check failed: %s", real)
        return OpenAIHealthResponse(
            status  = "error",
            key_set = True,
            message = f"Key is set but OpenAI call failed: {real}",
        )