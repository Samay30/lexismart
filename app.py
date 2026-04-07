"""
app.py — LexiSmart FastAPI application entry point.

Run locally:
    uvicorn app:app --reload --port 8000

On Render (start command):
    uvicorn app:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.pipeline import router as pipeline_router
from routers.health import router as health_router

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="LexiSmart — Pareto Readability Engine",
    description=(
        "Readability-controlled summarisation using multi-objective "
        "Pareto optimisation. Maximises Flesch Reading Ease subject to "
        "a semantic similarity constraint."
    ),
    version="2.0.0",
)

# ── CORS ──────────────────────────────────────────────────────
# The /v1/generate and /v1/embed endpoints are called directly from
# the user's browser (Netlify), so the Netlify origin must be listed.
# Do NOT use allow_origins=["*"] — that would expose the proxied
# OpenAI calls to any origin.
ALLOWED_ORIGINS = [
    "https://lexismart-v2.netlify.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

# ── Routers ───────────────────────────────────────────────────
# Health:  GET  /api/health
app.include_router(health_router, prefix="/api", tags=["health"])

# Pipeline (no prefix — routes define their own paths):
#   POST /v1/generate   ← frontend calls this for text generation
#   POST /v1/embed      ← frontend calls this for embeddings
#   POST /api/run       ← optional full server-side pipeline
#   POST /api/score     ← live readability scoring (no OpenAI call)
app.include_router(pipeline_router, tags=["pipeline"])