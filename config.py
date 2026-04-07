"""
config.py — Central configuration for the LexiSmart pipeline.

The OpenAI API key is loaded exclusively from the environment.
It is NEVER accepted from the client.

Set it on Render:
    Dashboard → lexismart service → Environment → OPENAI_API_KEY = sk-proj-...
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenAI — server-side only, never from the client ─────
    openai_api_key:     str   = ""              # required in production
    openai_model:       str   = "gpt-4o"
    openai_embed_model: str   = "text-embedding-3-small"
    openai_max_tokens:  int   = 512
    openai_retry_attempts: int = 3
    openai_retry_delay_ms: int = 600

    # ── Pipeline defaults ─────────────────────────────────────
    default_n_candidates:   int   = 8
    default_tau:            float = 0.80
    default_target_fre:     float = 72.0
    default_fre_band_high:  float = 90.0
    default_max_iterations: int   = 3
    default_temperature:    float = 0.75

    # ── Readability weights ────────────────────────────────────
    weight_fre:  float = 0.35
    weight_fkgl: float = 0.50
    weight_ttr:  float = 0.15

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()