"""
hallucination_check.py — Citation grounding verifier.

Checks whether claims in the LLM output are grounded in the provided context.
Uses sentence-level Jaccard overlap as the grounding signal.

For production: replace with a dedicated NLI model or GPT-4 fact-check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from scorer.local_scorer import _tokenize


@dataclass
class GroundingResult:
    is_grounded: bool
    grounded_sentences: int
    total_sentences: int
    grounding_ratio: float
    ungrounded: list[str] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]


def check_grounding(
    output_text: str,
    context: str,
    threshold: float = 0.20,
    min_ratio: float = 0.60,
) -> GroundingResult:
    """
    Verify that output sentences are grounded in context.

    Args:
        output_text: LLM-generated output.
        context: Source context / retrieved passages.
        threshold: Jaccard overlap required to consider a sentence grounded.
        min_ratio: Minimum fraction of sentences that must be grounded.

    Returns:
        GroundingResult with is_grounded flag.
    """
    sents = _sentences(output_text)
    if not sents:
        return GroundingResult(True, 0, 0, 1.0)

    ctx_tokens = set(_tokenize(context))
    grounded = 0
    ungrounded: list[str] = []

    for sent in sents:
        sent_tokens = set(_tokenize(sent))
        if not sent_tokens:
            continue
        overlap = len(sent_tokens & ctx_tokens) / len(sent_tokens)
        if overlap >= threshold:
            grounded += 1
        else:
            ungrounded.append(sent)

    ratio = grounded / len(sents)
    return GroundingResult(
        is_grounded=ratio >= min_ratio,
        grounded_sentences=grounded,
        total_sentences=len(sents),
        grounding_ratio=round(ratio, 4),
        ungrounded=ungrounded,
    )
