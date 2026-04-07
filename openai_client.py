"""
backend/openai_client.py — OpenAI API wrapper with retry logic.

Provides:
    AsyncOpenAIClient
        .complete(system, user, temperature)  → str
        .embed(text)                          → list[float]
    compute_similarity(vec_a, vec_b)          → float  (cosine)
"""

from __future__ import annotations
import asyncio
import logging
import math
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────

def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Cosine similarity between two dense float vectors.

    f2(s) = Sim(s, s_ref) = (a · b) / (‖a‖ × ‖b‖)

    Returns a value in [0.0, 1.0]:
        1.0 → identical direction (same meaning)
        0.0 → orthogonal (unrelated)

    Parameters
    ----------
    vec_a, vec_b : list[float]
        Embedding vectors of equal length.
    """
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


# ─────────────────────────────────────────────────────────────
# ASYNC OPENAI CLIENT
# ─────────────────────────────────────────────────────────────

class AsyncOpenAIClient:
    """
    Lightweight async OpenAI API client built on httpx.
    Supports GPT-4o chat completions and text-embedding-3-small.

    Parameters
    ----------
    api_key : str
        Your OpenAI API key (sk-proj-…).
    model : str
        Chat completion model (default: "gpt-4o").
    embed_model : str
        Embedding model (default: "text-embedding-3-small").
    max_tokens : int
        Maximum tokens in completion response.
    retry_attempts : int
        Number of retry attempts on transient errors.
    retry_delay_ms : int
        Base delay between retries in milliseconds (multiplied by attempt index).
    """

    BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str,
        model: str            = "gpt-4o",
        embed_model: str      = "text-embedding-3-small",
        max_tokens: int       = 512,
        retry_attempts: int   = 3,
        retry_delay_ms: int   = 600,
    ) -> None:
        self.api_key        = api_key
        self.model          = model
        self.embed_model    = embed_model
        self.max_tokens     = max_tokens
        self.retry_attempts = retry_attempts
        self.retry_delay_ms = retry_delay_ms

        self._headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    # ── Internal retry wrapper ──────────────────────────────

    async def _with_retry(self, coro_factory, label: str):
        """
        Execute *coro_factory()* up to retry_attempts times.
        Uses linear back-off: delay = retry_delay_ms × attempt_number.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.retry_attempts + 1):
            try:
                return await coro_factory()
            except Exception as exc:
                last_error = exc
                if attempt == self.retry_attempts:
                    break
                delay = self.retry_delay_ms * (attempt + 1) / 1000.0
                logger.warning(
                    "Retry %d/%d for [%s]: %s — waiting %.1fs",
                    attempt + 1, self.retry_attempts, label, exc, delay,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(f"[{label}] failed after {self.retry_attempts} retries: {last_error}")

    # ── Chat completion ─────────────────────────────────────

    async def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.75,
    ) -> str:
        """
        Call the GPT-4o chat completions endpoint.

        Parameters
        ----------
        system : str
            System role instruction.
        user : str
            User turn content.
        temperature : float
            Sampling temperature (0.0 – 1.4).

        Returns
        -------
        str
            The assistant's reply, stripped of leading/trailing whitespace.
        """
        async def _call():
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._headers,
                    json={
                        "model":       self.model,
                        "messages":    [
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user},
                        ],
                        "temperature": temperature,
                        "max_tokens":  self.max_tokens,
                        "n":           1,
                    },
                )
                self._raise_for_status(resp, "chat/completions")
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                )
                if not content:
                    raise RuntimeError("Empty response from model")
                return content.strip()

        return await self._with_retry(_call, "gpt-complete")

    # ── Embeddings ──────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """
        Embed *text* using text-embedding-3-small.

        Input is truncated at 8 000 characters to stay within token limits.

        Returns
        -------
        list[float]
            Dense embedding vector (1 536 dimensions).
        """
        truncated = text[:8000]

        async def _call():
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.BASE_URL}/embeddings",
                    headers=self._headers,
                    json={
                        "model": self.embed_model,
                        "input": truncated,
                    },
                )
                self._raise_for_status(resp, "embeddings")
                data = resp.json()
                return data["data"][0]["embedding"]

        return await self._with_retry(_call, "embed")

    # ── Error handling ──────────────────────────────────────

    @staticmethod
    def _raise_for_status(resp: httpx.Response, endpoint: str) -> None:
        """Raise a descriptive RuntimeError for non-2xx responses."""
        if resp.is_success:
            return
        try:
            body = resp.json()
            msg  = body.get("error", {}).get("message", f"HTTP {resp.status_code}")
        except Exception:
            msg = f"HTTP {resp.status_code}"

        if resp.status_code == 401:
            msg = "Invalid API key — check at platform.openai.com/api-keys"
        elif resp.status_code == 429:
            msg = "Rate limited / quota exceeded — check credits at platform.openai.com/usage"

        raise RuntimeError(f"[{endpoint}] {msg}")