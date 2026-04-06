"""
backend/pareto.py — Multi-objective Pareto optimisation.

Mathematical formulation
------------------------
Objectives (maximise both simultaneously):
    f1(s) = FRE(s)           — Flesch Reading Ease
    f2(s) = Sim(s, s_ref)   — cosine semantic similarity

Dominance relation:
    s_i dominates s_j  ⟺
        f1(s_i) ≥ f1(s_j)  AND  f2(s_i) ≥ f2(s_j)
        AND  (f1(s_i) > f1(s_j)  OR  f2(s_i) > f2(s_j))

Pareto frontier:
    P* = { s_i : ∄ s_j such that s_j dominates s_i }

Constrained optimisation problem:
    maximise   FRE(s)
    subject to Sim(s, s_ref) ≥ τ

If the feasible set { s : Sim(s) ≥ τ } is empty, fall back to argmax score.
"""

from __future__ import annotations
import logging
from typing import Optional

from backend.schemas import SummaryResult
from backend.metrics import compute_readability

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# BAND SCORE
# ─────────────────────────────────────────────────────────────

def band_score(
    fre:  float,
    sim:  float,
    fl:   float,
    fh:   float,
    tau:  float,
) -> float:
    """
    Composite band-proximity score ∈ [0, 1].

    Measures how close a candidate is to the (FRE × similarity) target band.
    Used as a tiebreaker and for colour coding in the frontend.

    Components:
        fre_prox — proximity to [FL, FH] window (1.0 if inside, decays outside)
        sim_prox — scaled distance above τ

    Returns 0.0 if sim < τ or fre < 40 (clearly unusable).

    Parameters
    ----------
    fre  : Flesch Reading Ease of the candidate
    sim  : cosine similarity of the candidate to the source
    fl   : lower bound of FRE target band
    fh   : upper bound of FRE target band
    tau  : similarity threshold τ
    """
    if sim < tau or fre < 40.0:
        return 0.0

    mid  = (fl + fh) / 2.0
    half = (fh - fl) / 2.0
    dist = abs(fre - mid)

    if dist <= half:
        fre_prox = 1.0
    else:
        fre_prox = max(0.0, 1.0 - (dist - half) / (half * 2.0))

    sim_prox = min(1.0, max(0.0, (sim - tau) / 0.15 + 0.5))

    return min(1.0, 0.6 * fre_prox + 0.4 * sim_prox)


# ─────────────────────────────────────────────────────────────
# CANDIDATE FACTORY
# ─────────────────────────────────────────────────────────────

def make_candidate(
    candidate_id: str,
    iteration:    int,
    text:         str,
    similarity:   float,
    tau:          float,
    fl:           float,
    fh:           float,
    w_fre:        float = 0.35,
    w_fkgl:       float = 0.50,
    w_ttr:        float = 0.15,
    action:       Optional[str] = None,
) -> SummaryResult:
    """
    Build a fully-scored SummaryResult from raw text + similarity value.

    Computes all readability metrics, the band score, and the in_band flag.

    Parameters
    ----------
    candidate_id : str   Unique identifier, e.g. "I1C3" (iter 1, candidate 3)
    iteration    : int   Which pipeline iteration generated this candidate
    text         : str   The candidate summary text
    similarity   : float Cosine similarity to the source embedding [0, 1]
    tau          : float Similarity threshold τ
    fl           : float FRE target band lower bound
    fh           : float FRE target band upper bound
    w_fre/fkgl/ttr       Readability metric weights
    action       : str   Optional refinement action label for debugging
    """
    rd = compute_readability(text, w_fre=w_fre, w_fkgl=w_fkgl, w_ttr=w_ttr)

    in_band = (rd.fre >= fl) and (rd.fre <= fh) and (similarity >= tau)
    score   = band_score(rd.fre, similarity, fl, fh, tau)

    return SummaryResult(
        id          = candidate_id,
        iteration   = iteration,
        text        = text,
        fre         = rd.fre,
        fkgl        = rd.fkgl,
        ttr         = rd.ttr,
        composite   = rd.composite,
        similarity  = round(similarity, 6),
        score       = round(score, 6),
        in_band     = in_band,
        action      = action,
    )


# ─────────────────────────────────────────────────────────────
# PARETO FRONTIER EXTRACTION
# ─────────────────────────────────────────────────────────────

def pareto_filter(candidates: list[SummaryResult]) -> list[SummaryResult]:
    """
    Extract the Pareto-optimal subset from *candidates*.

    Algorithm: naïve O(n²) pairwise domination check.
    For practical n (≤ 40 per run) this is fast and exact.

    Side effects:
        Sets .dominated = True  on every dominated candidate.
        Sets .is_pareto = True  on every non-dominated candidate.

    Returns
    -------
    list[SummaryResult]
        The Pareto frontier (non-dominated subset), unordered.
    """
    n = len(candidates)

    # Reset flags
    for c in candidates:
        c.dominated = False
        c.is_pareto = False

    # Pairwise dominance test
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ci, cj = candidates[i], candidates[j]

            # Does cj dominate ci?
            #   fre_j ≥ fre_i  AND  sim_j ≥ sim_i
            #   AND  (fre_j > fre_i  OR  sim_j > sim_i)
            j_dominates_i = (
                cj.fre >= ci.fre and
                cj.similarity >= ci.similarity and
                (cj.fre > ci.fre or cj.similarity > ci.similarity)
            )

            if j_dominates_i:
                ci.dominated = True
                break   # ci is dominated; no need to check further

    # Collect non-dominated set and tag
    frontier = [c for c in candidates if not c.dominated]
    for c in frontier:
        c.is_pareto = True

    logger.info(
        "Pareto frontier: %d / %d candidates are non-dominated",
        len(frontier), n,
    )
    return frontier


# ─────────────────────────────────────────────────────────────
# CONSTRAINED SELECTION
# ─────────────────────────────────────────────────────────────

def constrained_select(
    candidates: list[SummaryResult],
    tau: float,
) -> SummaryResult:
    """
    Solve the constrained optimisation problem:

        maximise   FRE(s)
        subject to Sim(s, s_ref) ≥ τ

    Step 1: Build the feasible set F = { s : Sim(s) ≥ τ }.
    Step 2: If F is non-empty → return argmax_{s ∈ F} FRE(s).
    Step 3: Else (no feasible candidate) → fallback to argmax score.

    Parameters
    ----------
    candidates : list[SummaryResult]
        All generated candidates (across all iterations).
    tau : float
        Similarity threshold τ.

    Returns
    -------
    SummaryResult
        The selected optimal candidate.
    """
    # Feasible set: sim ≥ τ
    feasible = [c for c in candidates if c.similarity >= tau]

    if feasible:
        # Constrained solution: argmax FRE among feasible candidates
        best = max(feasible, key=lambda c: c.fre)
        logger.info(
            "Constrained select: %d feasible candidates, best FRE=%.1f Sim=%.3f",
            len(feasible), best.fre, best.similarity,
        )
        return best

    # Fallback: no candidate satisfies τ — use argmax score
    logger.warning(
        "No candidate satisfies τ=%.2f — falling back to argmax score", tau
    )
    return max(candidates, key=lambda c: c.score)