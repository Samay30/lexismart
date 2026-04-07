"""
schemas.py — Pydantic request / response models.

Key change from v1: api_key is removed from ALL request models.
The server reads its own OPENAI_API_KEY from the environment (config.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel, Field


# ── /v1/generate request & response ──────────────────────────

class GenerateRequest(BaseModel):
    """Body for POST /v1/generate — sent by the frontend."""
    system:      str   = Field(..., min_length=1)
    user:        str   = Field(..., min_length=1)
    temperature: float = Field(0.75, ge=0.1, le=1.4)


class GenerateResponse(BaseModel):
    text: str


# ── /v1/embed request & response ─────────────────────────────

class EmbedRequest(BaseModel):
    """Body for POST /v1/embed — sent by the frontend."""
    text: str = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    embedding: list[float]


# ── /api/run request ──────────────────────────────────────────

class PipelineRequest(BaseModel):
    """
    Payload for POST /api/run (full server-side pipeline).
    No api_key field — key comes from environment.
    """
    source_text: str = Field(..., min_length=10)

    n_candidates:   int   = Field(8,    ge=3,  le=20)
    tau:            float = Field(0.80, ge=0.5, le=0.99)
    target_fre:     float = Field(72.0, ge=20,  le=100)
    fre_band_high:  float = Field(90.0, ge=50,  le=100)
    max_iterations: int   = Field(3,    ge=1,   le=5)
    temperature:    float = Field(0.75, ge=0.1, le=1.4)


# ── Internal candidate dataclass ──────────────────────────────

@dataclass
class SummaryResult:
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
    dominated:   bool         = False
    is_pareto:   bool         = False
    is_selected: bool         = False
    action:      Optional[str] = None


# ── /api/run response ─────────────────────────────────────────

class CandidateOut(BaseModel):
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
    best:             CandidateOut
    all_candidates:   list[CandidateOut]
    pareto_frontier:  list[CandidateOut]
    total_candidates: int
    pareto_size:      int
    constraint_met:   bool
    target_fre_met:   bool
    iterations_run:   int
    log:              list[dict]


class HealthResponse(BaseModel):
    status:  str
    version: str