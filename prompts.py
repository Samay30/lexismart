"""
backend/prompts.py — All GPT-4o prompt templates used by the pipeline.

Keeping prompts in one place makes them easy to iterate on without
touching orchestration logic.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────

SYSTEM_GENERATE = (
    "You are a plain-language summarizer.\n"
    "Rewrite the text so it is VERY easy to read (target Flesch Reading Ease ≥ 80).\n"
    "\n"
    "STRICT RULES:\n"
    "1. Use short sentences (MAX 10–12 words per sentence, ideally 8–10)\n"
    "2. Each sentence must contain ONLY ONE idea\n"
    "3. Break all long sentences into multiple short ones\n"
    "4. Use simple, common words (avoid words with more than 2 syllables)\n"
    "5. Replace formal words with simpler ones (e.g., \"demonstrate\" → \"show\")\n"
    "6. Use active voice only\n"
    "7. Preserve ALL facts, names, numbers, and meaning\n"
    "\n"
    "IMPORTANT:\n"
    "- You MUST rewrite structure, not lightly edit\n"
    "- Do NOT keep original sentence structure if it is complex\n"
    "\n"
    "Output ONLY the rewritten text."
)

SYSTEM_REFINE = (
    "You are improving readability further.\n"
    "\n"
    "Rewrite the text to make it EVEN easier to read.\n"
    "\n"
    "Focus on:\n"
    "- Shorter sentences\n"
    "- Simpler words\n"
    "- Breaking complex ideas into multiple sentences\n"
    "\n"
    "DO NOT remove any information.\n"
    "You are allowed to fully restructure the text.\n"
    "\n"
    "Output only the improved version."
)

# ─────────────────────────────────────────────────────────────
# LEAD-IN PHRASES FOR GENERATION DIVERSITY
# Cycled across N candidates so each call has a slightly different framing.
# ─────────────────────────────────────────────────────────────

_LEADS = [
    "Summarise clearly:",
    "Rewrite for clarity:",
    "Simplify this text:",
    "Plain-language rewrite:",
    "Easy-read version:",
]


# ─────────────────────────────────────────────────────────────
# USER PROMPT FACTORIES
# ─────────────────────────────────────────────────────────────

def prompt_generate(source: str, variation: int) -> str:
    """
    Generation prompt.
    Rotates through _LEADS for surface diversity across N candidates.

    Parameters
    ----------
    source    : str  The original source text.
    variation : int  Candidate index (0-based) used to select a lead phrase.
    """
    lead = _LEADS[variation % len(_LEADS)]
    return f"{lead}\n\n{source}"


def prompt_fidelity(text: str, similarity: float, source: str, tau: float) -> str:
    """
    Fidelity repair prompt — used when similarity < τ.
    Instructs the model to add back missing facts from the source.
    """
    return (
        f"Current similarity is {similarity:.3f} (needs ≥ {tau:.2f}).\n"
        "Add back all missing facts, names, numbers, causal links from the original.\n"
        "Keep sentences ≤ 15 words. Use simple words.\n"
        "\n"
        "Current text:\n"
        f"{text}\n"
        "\n"
        "Original source:\n"
        f"{source}"
    )


def prompt_split(text: str, fre: float, target_fre: float) -> str:
    """
    Sentence-split prompt — used when FRE is below target.
    Splits long sentences at conjunctions without changing vocabulary.
    """
    return (
        f"Readability score is {fre:.1f} (target ≥ {target_fre:.0f}).\n"
        "Split every sentence longer than 18 words at conjunctions "
        "(and, but, so, because, which).\n"
        "Do NOT change any words, vocabulary, or facts.\n"
        "\n"
        "Text:\n"
        f"{text}"
    )


def prompt_vocab(text: str, fre: float, target_fre: float) -> str:
    """
    Vocabulary simplification prompt — used when FRE is below target.
    Replaces polysyllabic words with simpler equivalents.
    """
    return (
        f"Readability score is {fre:.1f} (target ≥ {target_fre:.0f}).\n"
        "Replace every word with more than 2 syllables with a simpler equivalent.\n"
        "Keep sentence structure and all facts intact.\n"
        "\n"
        "Text:\n"
        f"{text}"
    )
