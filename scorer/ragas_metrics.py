"""
ragas_metrics.py — RAGAS-style retrieval evaluation metrics (pure Python, no RAGAS dependency).

Implements:
  - Faithfulness:  fraction of output claims grounded in context
  - Context Relevancy: fraction of context sentences relevant to query
  - Recall@K:  fraction of relevant docs retrieved in top-K
  - Precision@1: is the top result relevant?
  - MRR (Mean Reciprocal Rank)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .local_scorer import _tokenize


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RAGASResult:
    faithfulness: float = 0.0        # [0,1]
    context_relevancy: float = 0.0   # [0,1]
    recall_at_k: float = 0.0         # [0,1]
    precision_at_1: float = 0.0      # 0 or 1
    mrr: float = 0.0                 # [0,1]


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

def _sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def faithfulness(output_text: str, context: str, overlap_threshold: float = 0.20) -> float:
    """
    Fraction of output sentences that have Jaccard overlap ≥ threshold with context.
    A simple but fast approximation of RAGAS faithfulness.
    """
    sentences = _sentence_split(output_text)
    if not sentences:
        return 0.0

    ctx_tokens = set(_tokenize(context))
    grounded = 0
    for sent in sentences:
        sent_tokens = set(_tokenize(sent))
        if not sent_tokens:
            continue
        overlap = len(sent_tokens & ctx_tokens) / len(sent_tokens)
        if overlap >= overlap_threshold:
            grounded += 1

    return round(grounded / len(sentences), 4)


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "what", "which",
    "who", "whom", "this", "that", "these", "those", "it", "its", "as",
}


def context_relevancy(query: str, context: str, threshold: float = 0.15) -> float:
    """
    Fraction of context sentences that share substantive tokens with the query.
    """
    sentences = _sentence_split(context)
    if not sentences:
        return 0.0

    raw_query = set(_tokenize(query))
    content_query = raw_query - _STOPWORDS
    query_tokens = content_query if content_query else raw_query

    relevant = sum(
        1
        for s in sentences
        if len((set(_tokenize(s)) - _STOPWORDS) & query_tokens) / max(len(query_tokens), 1) >= threshold
    )
    return round(relevant / len(sentences), 4)


# ---------------------------------------------------------------------------
# Recall@K / Precision@1 / MRR
# (for ranked retrieval evaluation)
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    """Fraction of relevant docs found in top-K retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return round(len(top_k & relevant_ids) / len(relevant_ids), 4)


def precision_at_1(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Is the top result relevant?"""
    if not retrieved_ids:
        return 0.0
    return 1.0 if retrieved_ids[0] in relevant_ids else 0.0


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """MRR — reciprocal rank of the first relevant result."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return round(1.0 / rank, 4)
    return 0.0


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def evaluate_rag(
    query: str,
    output_text: str,
    context: str,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> RAGASResult:
    """Run all RAGAS-style metrics in one call."""
    return RAGASResult(
        faithfulness=faithfulness(output_text, context),
        context_relevancy=context_relevancy(query, context),
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
        precision_at_1=precision_at_1(retrieved_ids, relevant_ids),
        mrr=mean_reciprocal_rank(retrieved_ids, relevant_ids),
    )
