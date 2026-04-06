"""
backend/config.py — Central configuration for the LexiSmart pipeline.

All pipeline hyper-parameters live here. Can be overridden per-request
via the PipelineConfig Pydantic model (see schemas.py).
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Create a .env file in the project root to override defaults.

    Example .env:
        OPENAI_API_KEY=sk-proj-...
        DEFAULT_N_CANDIDATES=8
    """

    # ── OpenAI ────────────────────────────────────────────────
    openai_api_key: str = ""                # can also be passed per-request
    openai_model: str = "gpt-4o"
    openai_embed_model: str = "text-embedding-3-small"
    openai_max_tokens: int = 512
    openai_retry_attempts: int = 3
    openai_retry_delay_ms: int = 600        # base delay; multiplied by attempt

    # ── Pipeline defaults ─────────────────────────────────────
    default_n_candidates: int = 8           # number of candidates per iteration
    default_tau: float = 0.80              # similarity threshold τ
    default_target_fre: float = 72.0       # target Flesch Reading Ease
    default_fre_band_high: float = 90.0    # upper bound of FRE target band
    default_max_iterations: int = 3        # refinement loop iterations
    default_temperature: float = 0.75      # base GPT-4o sampling temperature

    # ── Readability weights ────────────────────────────────────
    # Composite = w_fre*FRE + w_fkgl*FKGL + w_ttr*TTR
    weight_fre:  float = 0.35
    weight_fkgl: float = 0.50
    weight_ttr:  float = 0.15

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
