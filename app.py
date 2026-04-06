"""
app.py — LexiSmart FastAPI application entry point.

Run with:
    uvicorn backend.app:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.routers import pipeline, health

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="LexiSmart — Pareto Readability Engine",
    description=(
        "Readability-controlled summarisation using multi-objective "
        "Pareto optimisation. Maximises Flesch Reading Ease subject to "
        "a semantic similarity constraint."
    ),
    version="2.0.0",
)

# ── CORS (allow frontend dev server on any port) ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(pipeline.router, prefix="/api", tags=["pipeline"])

# ── Static files ─────────────────────────────────────────────
FRONTEND = Path("index.html").parent.parent / "frontend"

# ── Serve index.html at root ──────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_root():
    return FileResponse(FRONTEND / "templates" / "index.html")
