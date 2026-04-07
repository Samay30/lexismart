"""
backend/routers/pipeline.py — FastAPI router for the pipeline endpoints.

Endpoints
---------
POST /api/run
    Run the full Pareto summarisation pipeline.
    Body: PipelineRequest (JSON)
    Response: PipelineResponse (JSON)

POST /api/score
    Score a single text (FRE + readability breakdown).
    Useful for the live source-stats display.
    Body: { "text": "..." }
    Response: readability metrics dict
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from schemas import PipelineRequest, PipelineResponse
from pipeline import run_pipeline
from metrics import compute_readability, compute_fkgl_raw

router = APIRouter()


# ── Run pipeline ──────────────────────────────────────────────

@router.post("/run", response_model=PipelineResponse)
async def run(request: PipelineRequest) -> PipelineResponse:
    """
    Execute the full readability-controlled summarisation pipeline.

    1. Embed source text
    2. Generate N candidate summaries (GPT-4o, temperature-varied)
    3. Compute FRE + cosine similarity for each candidate
    4. Extract Pareto frontier
    5. Constrained selection: argmax FRE | Sim ≥ τ
    6. Iterative refinement loop
    7. Return best summary + all candidates + Pareto frontier + log
    """
    try:
        result = await run_pipeline(request)
        return result
    except RuntimeError as exc:
        # Surface clean API errors (invalid key, quota, etc.)
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ── Single-text scoring ───────────────────────────────────────

class ScoreRequest(BaseModel):
    text: str


@router.post("/score")
async def score_text(body: ScoreRequest) -> dict:
    """
    Compute readability metrics for a single text.
    Used by the frontend for the live source-stats bar.
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
