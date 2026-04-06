"""
backend/metrics.py — Readability metrics implemented from scratch.

Formulas
--------
Flesch Reading Ease (FRE):
    FRE = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)

Flesch-Kincaid Grade Level (FKGL):
    FKGL = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59

Type-Token Ratio (TTR):
    TTR = unique_tokens / total_tokens
"""

from __future__ import annotations
import re
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────
# DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReadabilityScores:
    composite: float
    fre: float
    fkgl: float
    fkgl_raw: float
    ttr: float
    words: int
    sentences: int
    syllables: int


# ─────────────────────────────────────────────────────────────
# SYLLABLE ESTIMATION (IMPROVED)
# ─────────────────────────────────────────────────────────────

_VOWELS = frozenset("aeiouy")


def count_syllables_word(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 2:
        return 1

    count = len(re.findall(r"[aeiouy]+", w))

    # silent 'e'
    if w.endswith("e") and not w.endswith("le") and count > 1:
        count -= 1

    # -le ending
    if len(w) > 2 and w.endswith("le") and w[-3] not in _VOWELS:
        count += 1

    # smarter -ed handling
    if w.endswith("ed") and not re.search(r"[td]ed$", w):
        count = max(1, count - 1)

    # smarter -es handling
    if w.endswith("es") and not re.search(r"[sxz]es$|[^aeiou]hes$", w):
        count = max(1, count - 1)

    return max(1, count)


# ─────────────────────────────────────────────────────────────
# TEXT STATS (FIXED SENTENCE COUNTING)
# ─────────────────────────────────────────────────────────────

def text_stats(text: str) -> tuple[int, int, int]:
    words = re.findall(r"\b[a-zA-Z']+\b", text)
    word_count = max(1, len(words))

    syllable_count = sum(count_syllables_word(w) for w in words)

    # FIXED: no dropping short sentences
    parts = [p for p in re.split(r"[.!?]+", text) if p.strip()]
    sentence_count = max(1, len(parts))

    return word_count, syllable_count, sentence_count


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_fre(w: int, sy: int, s: int) -> float:
    raw = 206.835 - 1.015 * (w / s) - 84.6 * (sy / w)
    return max(0.0, min(100.0, raw))


def compute_fkgl_raw(w: int, sy: int, s: int) -> float:
    return 0.39 * (w / s) + 11.8 * (sy / w) - 15.59


def compute_fkgl_norm(fkgl_raw: float) -> float:
    return max(0.0, min(100.0, 100.0 - fkgl_raw * 10.0))


def compute_ttr(text: str) -> float:
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) < 5:
        return 50.0

    ratio = len(set(tokens)) / len(tokens)
    return max(0.0, min(100.0, 100.0 - ratio * 85.0))


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def compute_readability(
    text: str,
    w_fre: float = 0.7,
    w_fkgl: float = 0.3,
    w_ttr: float = 0.0,
) -> ReadabilityScores:
    """
    Cleaner composite:
    - FRE is primary signal
    - FKGL secondary
    - TTR optional (disabled by default)
    """

    word_count, syllable_count, sentence_count = text_stats(text)

    fre = compute_fre(word_count, syllable_count, sentence_count)
    fkgl_raw = compute_fkgl_raw(word_count, syllable_count, sentence_count)
    fkgl = compute_fkgl_norm(fkgl_raw)
    ttr = compute_ttr(text)

    composite = w_fre * fre + w_fkgl * fkgl + w_ttr * ttr

    return ReadabilityScores(
        composite=round(composite, 4),
        fre=round(fre, 4),
        fkgl=round(fkgl, 4),
        fkgl_raw=round(fkgl_raw, 4),
        ttr=round(ttr, 4),
        words=word_count,
        sentences=sentence_count,
        syllables=syllable_count,
    )