"""
routers/pipeline.py — FastAPI router.

Endpoints
---------
POST /v1/generate   — called directly by the frontend (generate text)
POST /v1/embed      — called directly by the frontend (embed text)
POST /api/run       — full server-side pipeline (optional / future)
POST /api/score     — live readability scoring for source stats bar
"""

from fastapi import APIRouter, HTTPException

from schemas import (
    GenerateRequest, GenerateResponse,
    EmbedRequest, EmbedResponse,
    PipelineRequest, PipelineResponse,
)
from .pipeline_core import run_pipeline
from metrics import compute_readability, compute_fkgl_raw
from openai_client import AsyncOpenAIClient
from config import settings
from pydantic import BaseModel

router = APIRouter()


# ── Shared client factory ─────────────────────────────────────

def _get_client() -> AsyncOpenAIClient:
    """
    Return an AsyncOpenAIClient using the server-side API key.
    Raises a clean 503 if the key is not configured.
    """
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="Service not configured. OPENAI_API_KEY is missing on the server.",
        )
    return AsyncOpenAIClient(
        api_key        = settings.openai_api_key,
        model          = settings.openai_model,
        embed_model    = settings.openai_embed_model,
        max_tokens     = settings.openai_max_tokens,
        retry_attempts = settings.openai_retry_attempts,
        retry_delay_ms = settings.openai_retry_delay_ms,
    )


# ── POST /v1/generate ─────────────────────────────────────────

@router.post("/v1/generate", response_model=GenerateResponse)
async def v1_generate(body: GenerateRequest) -> GenerateResponse:
    """
    Generate text via GPT-4o.
    Called by the frontend's gpt() function.

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
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail="Service unavailable. Please try again shortly.")
    except Exception:
        raise HTTPException(status_code=500, detail="Service unavailable. Please try again shortly.")


# ── POST /v1/embed ────────────────────────────────────────────

@router.post("/v1/embed", response_model=EmbedResponse)
async def v1_embed(body: EmbedRequest) -> EmbedResponse:
    """
    Embed text via text-embedding-3-small.
    Called by the frontend's embed() function.

    Request:  { "text": str }
    Response: { "embedding": float[] }
    """
    client = _get_client()
    try:
        embedding = await client.embed(body.text)
        return EmbedResponse(embedding=embedding)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail="Service unavailable. Please try again shortly.")
    except Exception:
        raise HTTPException(status_code=500, detail="Service unavailable. Please try again shortly.")


# ── POST /api/run ─────────────────────────────────────────────

@router.post("/api/run", response_model=PipelineResponse)
async def run(request: PipelineRequest) -> PipelineResponse:
    """
    Execute the full server-side Pareto pipeline.
    Useful for batch processing or server-rendered results.
    """
    try:
        return await run_pipeline(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error.")


# ── POST /api/score ───────────────────────────────────────────

class ScoreRequest(BaseModel):
    text: str


@router.post("/api/score")
async def score_text(body: ScoreRequest) -> dict:
    """
    Compute readability metrics for a single text.
    Used by the frontend live source-stats bar.
    No OpenAI call — pure local computation.
    """
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    rd    = compute_readability(body.text)
    grade = compute_fkgl_raw(body.text)

    return {
        "composite":  rd.composite,
        "fre":        rd.fre,
        "fkgl":       rd.fkgl,
        "fkgl_raw":   grade,
        "ttr":        rd.ttr,
        "words":      rd.words,
        "sentences":  rd.sentences,
        "syllables":  rd.syllables,
    }