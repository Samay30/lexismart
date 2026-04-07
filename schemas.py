"""
backend/schemas.py — Pydantic request / response models.

All data that flows between the frontend and backend is typed here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel, Field


# ── /v1/generate ─────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Payload for POST /v1/generate (called directly by the frontend)."""
    system:      str   = Field(..., description="System prompt")
    user:        str   = Field(..., description="User prompt / source text")
    temperature: float = Field(0.75, ge=0.1, le=1.4)


class GenerateResponse(BaseModel):
    """Response for POST /v1/generate."""
    text: str


# ── /v1/embed ────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    """Payload for POST /v1/embed (called directly by the frontend)."""
    text: str = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    """Response for POST /v1/embed."""
    embedding: list[float]


# ── /api/run request ──────────────────────────────────────────

class PipelineRequest(BaseModel):
    """
    Payload sent by the frontend to POST /api/run.

    api_key is no longer accepted from the client — the key is loaded
    server-side from the OPENAI_API_KEY environment variable.
    """
    source_text: str = Field(..., min_length=10, description="Input document to summarise")

    # Pipeline hyper-parameters (all optional — fall back to config.py defaults)
    n_candidates:   int   = Field(8,    ge=3,  le=20)
    tau:            float = Field(0.80, ge=0.5, le=0.99, description="Similarity threshold τ")
    target_fre:     float = Field(72.0, ge=20, le=100,   description="Minimum FRE goal")
    fre_band_high:  float = Field(90.0, ge=50, le=100,   description="FRE band upper bound")
    max_iterations: int   = Field(3,    ge=1,  le=5)
    temperature:    float = Field(0.75, ge=0.1, le=1.4)


# ── Candidate (internal dataclass) ────────────────────────────

@dataclass
class SummaryResult:
    """
    Structured result for a single candidate summary.

    Fields mirror the mathematical problem:
      fre        → f1(s) = FRE(s)
      similarity → f2(s) = Sim(s, s_ref)
      dominated  → set True if another candidate dominates this one
      is_pareto  → True if on the Pareto frontier P*
      is_selected → True if chosen as the constrained optimum
    """
    id:          str
    iteration:   int
    text:        str

    # Readability metrics
    fre:         float        # Flesch Reading Ease         [0, 100]
    fkgl:        float        # Normalised FKGL score       [0, 100]
    ttr:         float        # Type-Token Ratio score      [0, 100]
    composite:   float        # Weighted composite          [0, 100]

    # Semantic similarity
    similarity:  float        # Cosine sim to source embedding [0, 1]

    # Derived
    score:       float        # Band proximity score        [0, 1]
    in_band:     bool         # fre ∈ [FL, FH] AND sim ≥ τ

    # Pareto flags (set by pareto.py)
    dominated:   bool = False
    is_pareto:   bool = False
    is_selected: bool = False

    # Debug info
    action:      Optional[str] = None   # refinement action applied, e.g. 'vocab-simplify'


# ── Response ──────────────────────────────────────────────────

class CandidateOut(BaseModel):
    """Serialisable version of SummaryResult for the API response."""
    id:          str
    iteration:   int
    text:        str
    fre:         float
    fkgl:        float
    ttr:         float
    composite:   float
    similarity:  float
    score:       float
    in_band:     bool
    dominated:   bool
    is_pareto:   bool
    is_selected: bool
    action:      Optional[str] = None


class PipelineResponse(BaseModel):
    """
    Full pipeline response returned to the frontend.
    """
    # The chosen optimal summary
    best: CandidateOut

    # All candidates generated (across all iterations)
    all_candidates: list[CandidateOut]

    # Pareto-optimal subset
    pareto_frontier: list[CandidateOut]

    # Summary stats
    total_candidates:   int
    pareto_size:        int
    constraint_met:     bool   # best.similarity >= tau
    target_fre_met:     bool   # best.fre >= target_fre
    iterations_run:     int

    # Timestamped log lines for the frontend log panel
    log: list[dict]            # [{time_s: float, message: str, level: str}]


class HealthResponse(BaseModel):
    status: str
    version: str