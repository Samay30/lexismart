"""
routers/pipeline.py — FastAPI router.

Endpoints
---------
POST /v1/generate   — called directly by the frontend (generate text)
POST /v1/embed      — called directly by the frontend (embed text)
POST /api/run       — full server-side pipeline
POST /api/score     — live readability scoring (no OpenAI call)
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from schemas import (
    GenerateRequest, GenerateResponse,
    EmbedRequest, EmbedResponse,
    PipelineRequest, PipelineResponse,
)
from pipeline import run_pipeline
from metrics import compute_readability, text_stats, compute_fkgl_raw
from openai_client import AsyncOpenAIClient
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Shared client factory ─────────────────────────────────────

def _get_client() -> AsyncOpenAIClient:
    """
    Build an AsyncOpenAIClient from the server-side env key.
    Returns 503 immediately if OPENAI_API_KEY is not configured —
    no point retrying a missing key.
    """
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY is not set in environment")
        raise HTTPException(
            status_code=503,
            detail="Service not configured — contact the administrator.",
        )
    return AsyncOpenAIClient(
        api_key        = settings.openai_api_key,
        model          = settings.openai_model,
        embed_model    = settings.openai_embed_model,
        max_tokens     = settings.openai_max_tokens,
        retry_attempts = settings.openai_retry_attempts,
        retry_delay_ms = settings.openai_retry_delay_ms,
    )


def _handle_openai_error(exc: Exception, endpoint: str) -> HTTPException:
    """
    Log the real exception server-side (visible in Render logs),
    then return a sanitised HTTPException for the client.

    The raw OpenAI error message is included so the frontend can show
    actionable feedback (e.g. "Invalid API key", "Rate limited").
    Internal tracebacks are never sent to the client.
    """
    real_msg = str(exc)
    logger.exception("OpenAI call failed on %s: %s", endpoint, real_msg)

    # Pass through specific actionable messages; hide everything else
    PASSTHROUGH = ("Invalid API key", "Rate limited", "quota exceeded", "not set")
    client_msg = (
        real_msg if any(k in real_msg for k in PASSTHROUGH)
        else "Service unavailable — please try again shortly."
    )
    return HTTPException(status_code=502, detail=client_msg)


# ── POST /v1/generate ─────────────────────────────────────────

@router.post("/v1/generate", response_model=GenerateResponse)
async def v1_generate(body: GenerateRequest) -> GenerateResponse:
    """
    Generate text via GPT-4o.
    Request:  { "system": str, "user": str, "temperature": float }
    Response: { "text": str }
    """
    client = _get_client()
    try:
        text = await client.complete(
            system      = body.system,
            user        = body.user,
            temperature = body.temperature,
        )
        return GenerateResponse(text=text)
    except Exception as exc:
        raise _handle_openai_error(exc, "/v1/generate")


# ── POST /v1/embed ────────────────────────────────────────────

@router.post("/v1/embed", response_model=EmbedResponse)
async def v1_embed(body: EmbedRequest) -> EmbedResponse:
    """
    Embed text via text-embedding-3-small.
    Request:  { "text": str }
    Response: { "embedding": float[] }
    """
    client = _get_client()
    try:
        embedding = await client.embed(body.text)
        return EmbedResponse(embedding=embedding)
    except Exception as exc:
        raise _handle_openai_error(exc, "/v1/embed")


# ── POST /api/run ─────────────────────────────────────────────

@router.post("/api/run", response_model=PipelineResponse)
async def run(request: PipelineRequest) -> PipelineResponse:
    """Full server-side Pareto pipeline."""
    try:
        return await run_pipeline(request)
    except Exception as exc:
        raise _handle_openai_error(exc, "/api/run")


# ── POST /api/score ───────────────────────────────────────────

class ScoreRequest(BaseModel):
    text: str


@router.post("/api/score")
async def score_text(body: ScoreRequest) -> dict:
    """
    Compute readability metrics for a single text.
    Pure local computation — no OpenAI call.

    FIX: compute_fkgl_raw(w, sy, s) takes three ints, not a string.
    We call text_stats() first to get the raw counts.
    """
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    rd = compute_readability(body.text)

    # compute_fkgl_raw needs (word_count, syllable_count, sentence_count)
    # — these are available on the ReadabilityScores dataclass as .fkgl_raw
    # directly, so we just use that instead of calling it again.
    return {
        "composite":  rd.composite,
        "fre":        rd.fre,
        "fkgl":       rd.fkgl,
        "fkgl_raw":   rd.fkgl_raw,   # ← was: compute_fkgl_raw(body.text) — wrong!
        "ttr":        rd.ttr,
        "words":      rd.words,
        "sentences":  rd.sentences,
        "syllables":  rd.syllables,
    }
