"""
local_scorer.py — Rule-based fast scorer (0.013ms/probe)

Metrics:
  - Type-Token Ratio (TTR): vocabulary diversity
  - N-gram windowing: degenerate repetition detection
  - Jaccard overlap: circular output detection (input ≈ output)
  - Regex syntax markers: punctuation, quote balance, formatting

Extracted and generalized from CIRCLE's LightweightStudentEvaluator.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class LocalScore:
    ttr: float = 0.0
    ngram_repetition: float = 0.0      # lower = more repetitive
    jaccard_overlap: float = 0.0       # overlap with input; high = circular
    syntax_ok: bool = True
    composite: float = 0.0            # final [0,1] score
    flags: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.composite >= 0.70


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _ttr(tokens: list[str]) -> float:
    """Type-Token Ratio — vocabulary richness."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _ngram_repetition_score(tokens: list[str], n: int = 3, window: int = 40) -> float:
    """
    Slide a window of `window` tokens and measure fraction of unique n-grams.
    Score of 1.0 = fully diverse; 0.0 = completely repetitive.
    """
    if len(tokens) < n:
        return 1.0

    ngrams: list[tuple[str, ...]] = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i : i + n]))

    if not ngrams:
        return 1.0

    counts = Counter(ngrams)
    # fraction of unique n-grams
    unique_ratio = len(counts) / len(ngrams)
    return round(unique_ratio, 4)


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _syntax_check(text: str) -> tuple[bool, list[str]]:
    """Light regex checks for malformed outputs."""
    flags: list[str] = []
    ok = True

    # Unbalanced quotes
    single_quotes = text.count("'")
    double_quotes = text.count('"')
    if single_quotes % 2 != 0:
        flags.append("unbalanced_single_quotes")
        ok = False
    if double_quotes % 2 != 0:
        flags.append("unbalanced_double_quotes")
        ok = False

    # Ends without sentence-ending punctuation (allow JSON / lists)
    stripped = text.strip()
    if stripped and stripped[-1] not in ".!?}]\"'":
        flags.append("missing_terminal_punctuation")

    # Extremely short output
    if len(stripped.split()) < 3:
        flags.append("too_short")
        ok = False

    return ok, flags


def score(input_text: str, output_text: str) -> LocalScore:
    """
    Score a single (input, output) pair.

    Returns a LocalScore with composite ∈ [0, 1].
    Fast: <0.02ms per call on typical outputs.
    """
    stripped_out = output_text.strip() if output_text else ""
    if not stripped_out:
        return LocalScore(
            ttr=0.0,
            ngram_repetition=0.0,
            jaccard_overlap=0.0,
            syntax_ok=False,
            composite=0.0,
            flags=["empty_output", "too_short"],
        )

    inp_tokens = _tokenize(input_text)
    out_tokens = _tokenize(output_text)

    ttr = _ttr(out_tokens)
    ngram_rep = _ngram_repetition_score(out_tokens)
    jaccard = _jaccard(set(inp_tokens), set(out_tokens))
    syntax_ok, flags = _syntax_check(output_text)

    # Penalise high circular overlap
    circular_penalty = max(0.0, jaccard - 0.6)  # penalty kicks in above 0.6

    # Composite: weighted sum
    composite = (
        0.35 * ttr
        + 0.35 * ngram_rep
        + 0.20 * (1.0 - circular_penalty)
        + 0.10 * (1.0 if syntax_ok else 0.0)
    )
    composite = round(min(max(composite, 0.0), 1.0), 4)

    if jaccard > 0.6:
        flags.append("circular_output")

    return LocalScore(
        ttr=round(ttr, 4),
        ngram_repetition=round(ngram_rep, 4),
        jaccard_overlap=round(jaccard, 4),
        syntax_ok=syntax_ok,
        composite=composite,
        flags=flags,
    )


def batch_score(pairs: list[tuple[str, str]]) -> list[LocalScore]:
    """Score a batch of (input, output) pairs. Returns results in same order."""
    return [score(inp, out) for inp, out in pairs]
