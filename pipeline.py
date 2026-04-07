"""
pipeline.py — Readability-controlled summarisation pipeline.

Change from v1: client is initialised from settings.openai_api_key
(environment variable) instead of request.api_key.
"""

from __future__ import annotations
import logging
import time
from typing import Optional

from config import settings
from schemas import PipelineRequest, PipelineResponse, SummaryResult, CandidateOut
from metrics import compute_readability
from openai_client import AsyncOpenAIClient, compute_similarity
from pareto import make_candidate, pareto_filter, constrained_select
from prompts import (
    SYSTEM_GENERATE, SYSTEM_REFINE,
    prompt_generate, prompt_fidelity, prompt_split, prompt_vocab,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# PIPELINE LOGGER
# ─────────────────────────────────────────────────────────────

class PipelineLogger:
    def __init__(self) -> None:
        self._start = time.monotonic()
        self._lines: list[dict] = []

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._start, 3)

    def info(self, msg: str) -> None:
        self._lines.append({"time_s": self._elapsed(), "message": msg, "level": "info"})
        logger.info(msg)

    def ok(self, msg: str) -> None:
        self._lines.append({"time_s": self._elapsed(), "message": msg, "level": "ok"})
        logger.info(msg)

    def warn(self, msg: str) -> None:
        self._lines.append({"time_s": self._elapsed(), "message": msg, "level": "warn"})
        logger.warning(msg)

    def err(self, msg: str) -> None:
        self._lines.append({"time_s": self._elapsed(), "message": msg, "level": "err"})
        logger.error(msg)

    @property
    def lines(self) -> list[dict]:
        return self._lines


# ─────────────────────────────────────────────────────────────
# HELPERS — unchanged from v1
# ─────────────────────────────────────────────────────────────

async def generate_candidates(
    source, src_emb, n, base_temp, iteration, client, tau, fl, fh, log
) -> list[SummaryResult]:
    TEMP_OFFSETS = [0.0, +0.15, -0.10]
    candidates: list[SummaryResult] = []

    for i in range(n):
        temp = max(0.1, min(1.4, base_temp + TEMP_OFFSETS[i % 3]))
        log.info(f"Iteration {iteration} · generating candidate {i+1}/{n} (T={temp:.2f})…")

        text = await client.complete(
            system=SYSTEM_GENERATE,
            user=prompt_generate(source, i),
            temperature=temp,
        )
        emb = await client.embed(text)
        sim = compute_similarity(emb, src_emb)

        cand = make_candidate(
            candidate_id=f"I{iteration}C{i+1}",
            iteration=iteration,
            text=text,
            similarity=sim,
            tau=tau, fl=fl, fh=fh,
            w_fre=settings.weight_fre,
            w_fkgl=settings.weight_fkgl,
            w_ttr=settings.weight_ttr,
        )
        candidates.append(cand)

        level = log.ok if cand.in_band else log.info
        level(
            f"  → FRE={cand.fre:.1f}  Sim={cand.similarity:.3f}  "
            f"Score={cand.score:.3f}  inBand={cand.in_band}"
        )

    return candidates


async def refine_candidate(
    best, source, src_emb, step, tau, fl, fh, target_fre, client, log
) -> Optional[SummaryResult]:
    if best.similarity < tau:
        action = "fidelity-repair"
        user   = prompt_fidelity(best.text, best.similarity, source, tau)
        log.warn(f"Refine[{step}]: fidelity repair (sim={best.similarity:.3f} < τ={tau:.2f})")

    elif best.fre < target_fre:
        if step % 2 == 1:
            action = "sentence-split"
            user   = prompt_split(best.text, best.fre, target_fre)
        else:
            action = "vocab-simplify"
            user   = prompt_vocab(best.text, best.fre, target_fre)
        log.warn(f"Refine[{step}]: {action} (FRE={best.fre:.1f} < target={target_fre:.0f})")

    else:
        log.ok(f"Refine[{step}]: already meets FRE≥{target_fre} AND Sim≥{tau:.2f} — skipping")
        return None

    text = await client.complete(system=SYSTEM_REFINE, user=user, temperature=0.3)
    emb  = await client.embed(text)
    sim  = compute_similarity(emb, src_emb)

    cand = make_candidate(
        candidate_id=f"R{step}", iteration=step,
        text=text, similarity=sim,
        tau=tau, fl=fl, fh=fh,
        w_fre=settings.weight_fre,
        w_fkgl=settings.weight_fkgl,
        w_ttr=settings.weight_ttr,
        action=action,
    )
    log.info(f"  → {action}: FRE={cand.fre:.1f}  Sim={cand.similarity:.3f}  Score={cand.score:.3f}")
    return cand


def _to_out(c: SummaryResult) -> CandidateOut:
    return CandidateOut(
        id=c.id, iteration=c.iteration, text=c.text,
        fre=c.fre, fkgl=c.fkgl, ttr=c.ttr, composite=c.composite,
        similarity=c.similarity, score=c.score, in_band=c.in_band,
        dominated=c.dominated, is_pareto=c.is_pareto,
        is_selected=c.is_selected, action=c.action,
    )


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# Key change: client uses settings.openai_api_key — NOT request.api_key
# ─────────────────────────────────────────────────────────────

async def run_pipeline(request: PipelineRequest) -> PipelineResponse:
    log = PipelineLogger()

    n          = request.n_candidates
    tau        = request.tau
    fl         = request.target_fre
    fh         = request.fre_band_high
    max_iter   = request.max_iterations
    base_temp  = request.temperature
    target_fre = request.target_fre
    source     = request.source_text

    log.info(
        f"Pipeline start · N={n} τ={tau:.2f} targetFRE={target_fre:.0f} "
        f"maxIter={max_iter} T={base_temp:.2f}"
    )

    # ── Client uses server-side API key from environment ──────
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set on the server. "
            "Add it in Render → Environment variables."
        )

    client = AsyncOpenAIClient(
        api_key        = settings.openai_api_key,   # ← server env var only
        model          = settings.openai_model,
        embed_model    = settings.openai_embed_model,
        max_tokens     = settings.openai_max_tokens,
        retry_attempts = settings.openai_retry_attempts,
        retry_delay_ms = settings.openai_retry_delay_ms,
    )

    log.info("Embedding source text…")
    src_emb = await client.embed(source)
    log.ok(f"Source embedded ({len(src_emb)}-dim)")

    log.info(f"Generating {n} candidate summaries (iteration 1)…")
    all_candidates: list[SummaryResult] = await generate_candidates(
        source=source, src_emb=src_emb, n=n,
        base_temp=base_temp, iteration=1,
        client=client, tau=tau, fl=fl, fh=fh, log=log,
    )
    log.ok(f"Generation complete · {len(all_candidates)} candidates scored")

    log.info("Extracting Pareto frontier…")
    pareto_set = pareto_filter(all_candidates)

    log.info(f"Constrained selection: argmax FRE | sim ≥ τ={tau:.2f}…")
    best = constrained_select(all_candidates, tau)
    log.info(f"Initial best: FRE={best.fre:.1f}  Sim={best.similarity:.3f}  inBand={best.in_band}")

    iterations_run = 0
    for t in range(1, max_iter + 1):
        early_stop = best.similarity >= tau and best.fre >= target_fre
        log.info(
            f"--- Iteration {t}/{max_iter} · FRE={best.fre:.1f} "
            f"Sim={best.similarity:.3f} earlyStop={early_stop}"
        )
        if early_stop:
            log.ok(f"Early stop: FRE≥{target_fre:.0f} AND Sim≥{tau:.2f}")
            break

        refined = await refine_candidate(
            best=best, source=source, src_emb=src_emb,
            step=t, tau=tau, fl=fl, fh=fh, target_fre=target_fre,
            client=client, log=log,
        )
        iterations_run += 1
        if refined is None:
            break

        all_candidates.append(refined)
        pareto_set = pareto_filter(all_candidates)
        new_best = constrained_select(all_candidates, tau)
        if new_best.score >= best.score - 0.01:
            best = new_best

    log.info("Final constrained selection…")
    final = constrained_select(all_candidates, tau)
    final.is_selected = True

    log.ok(
        f"FINAL · FRE={final.fre:.1f}  Sim={final.similarity:.3f}  "
        f"Pareto={final.is_pareto}  inBand={final.in_band}"
    )
    log.ok(
        f"Pareto frontier size: {len(pareto_set)}  "
        f"· Total candidates evaluated: {len(all_candidates)}"
    )

    return PipelineResponse(
        best             = _to_out(final),
        all_candidates   = [_to_out(c) for c in all_candidates],
        pareto_frontier  = [_to_out(c) for c in pareto_set],
        total_candidates = len(all_candidates),
        pareto_size      = len(pareto_set),
        constraint_met   = final.similarity >= tau,
        target_fre_met   = final.fre >= target_fre,
        iterations_run   = iterations_run,
        log              = log.lines,
    )