"""
app.py — LexiSmart FastAPI application entry point.

Run with:
    uvicorn app:app --reload --port 8000

Frontend is served separately from Netlify (lexismart-v2.netlify.app).
This backend exposes:
    GET  /health
    POST /v1/generate
    POST /v1/embed
    POST /api/run      (full pipeline)
    POST /api/score    (single-text readability scoring)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import pipeline, health

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

# ── CORS ─────────────────────────────────────────────────────
# Allow the Netlify frontend and local dev only.
# Do NOT use allow_origins=["*"] — the API key lives server-side now.
ALLOWED_ORIGINS = [
    "https://lexismart-v2.netlify.app",
    "https://lexismart.netlify.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# ── Routers ──────────────────────────────────────────────────
app.include_router(health.router,   tags=["health"])
app.include_router(pipeline.router, tags=["pipeline"])