"""RAG evaluation metrics for retrieval, generation, and citations."""

from __future__ import annotations

import math
from collections.abc import Sequence

from rague.citations.models import CitedAnswer


def calculate_citation_rate(answer: CitedAnswer) -> float:
    """Return the share of claims that have at least one citation reference."""
    if not answer.claims:
        return 0.0

    cited_count = sum(1 for claim in answer.claims if claim.citation_refs)
    return cited_count / len(answer.claims)


def calculate_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Return the share of relevant documents among the first k retrieved."""
    if k <= 0:
        return 0.0

    relevant = set(relevant_ids)
    retrieved_top_k = retrieved_ids[:k]
    if not retrieved_top_k:
        return 0.0

    relevant_found = sum(1 for doc_id in retrieved_top_k if doc_id in relevant)
    return relevant_found / k


def calculate_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Return the share of relevant documents found in the first k retrieved."""
    if k <= 0 or not relevant_ids:
        return 0.0

    relevant = set(relevant_ids)
    retrieved_top_k = retrieved_ids[:k]
    relevant_found = sum(1 for doc_id in relevant if doc_id in retrieved_top_k)
    return relevant_found / len(relevant)


def calculate_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> float:
    """Return reciprocal rank of the first relevant document, or 0.0 if none."""
    relevant = set(relevant_ids)
    for index, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant:
            return 1.0 / (index + 1)
    return 0.0


def calculate_mrr(results: Sequence[dict[str, object]]) -> float:
    """Return mean reciprocal rank across per-query retrieval results."""
    if not results:
        return 0.0

    reciprocal_ranks: list[float] = []
    for result in results:
        retrieved = result.get("retrieved_docs", [])
        relevant = result.get("relevant_docs", [])
        if not isinstance(retrieved, list) or not isinstance(relevant, list):
            reciprocal_ranks.append(0.0)
            continue
        reciprocal_ranks.append(
            calculate_reciprocal_rank(retrieved, relevant)
        )

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Return NDCG@k with binary relevance for retrieved document ids."""
    if k <= 0:
        return 0.0

    relevant = set(relevant_ids)
    retrieved_top_k = retrieved_ids[:k]

    def _dcg(doc_ids: Sequence[str]) -> float:
        score = 0.0
        for index, doc_id in enumerate(doc_ids):
            if doc_id in relevant:
                score += 1.0 / math.log2(index + 2)
        return score

    dcg = _dcg(retrieved_top_k)
    if dcg == 0.0:
        return 0.0

    ideal_relevant = list(relevant)[:k]
    idcg = _dcg(ideal_relevant)
    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def calculate_answer_contains_score(
    answer_text: str,
    expected_substrings: Sequence[str] | None,
) -> float | None:
    """Return 1.0 when all expected substrings are present, else 0.0.

    Returns None when the case has no expected substrings to check.
    """
    if expected_substrings is None:
        return None

    normalized_answer = answer_text.casefold()
    if not expected_substrings:
        return 1.0

    for substring in expected_substrings:
        if substring.casefold() not in normalized_answer:
            return 0.0
    return 1.0


def calculate_citation_compliance(
    answer: CitedAnswer | None,
    *,
    should_cite: bool,
    min_citation_rate: float = 0.0,
) -> bool:
    """Return whether citation behavior matches the case expectation."""
    if not should_cite:
        if answer is None:
            return True
        return calculate_citation_rate(answer) == 0.0

    if answer is None:
        return False
    return calculate_citation_rate(answer) > min_citation_rate
