from __future__ import annotations

from langchain_core.documents import Document

from rague.citations import build_citation_context, build_cited_answer_from_claim_specs
from rague.evaluation.metrics import (
    calculate_answer_contains_score,
    calculate_citation_compliance,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)


def _page_doc(*, chunk_id: str) -> Document:
    return Document(
        page_content="sample",
        metadata={"chunk_id": chunk_id},
        id=chunk_id,
    )


def test_calculate_precision_at_k_counts_relevant_in_top_k() -> None:
    retrieved = ["doc-a", "doc-b", "doc-c"]
    relevant = ["doc-b", "doc-d"]

    assert calculate_precision_at_k(retrieved, relevant, k=2) == 0.5


def test_calculate_precision_at_k_zero_k_returns_zero() -> None:
    assert calculate_precision_at_k(["doc-a"], ["doc-a"], k=0) == 0.0


def test_calculate_precision_at_k_empty_retrieved_returns_zero() -> None:
    assert calculate_precision_at_k([], ["doc-a"], k=3) == 0.0


def test_calculate_recall_at_k_finds_fraction_of_relevant_docs() -> None:
    retrieved = ["doc-a", "doc-b", "doc-c"]
    relevant = ["doc-b", "doc-d"]

    assert calculate_recall_at_k(retrieved, relevant, k=2) == 0.5


def test_calculate_recall_at_k_empty_relevant_returns_zero() -> None:
    assert calculate_recall_at_k(["doc-a"], [], k=3) == 0.0


def test_calculate_recall_at_k_zero_k_returns_zero() -> None:
    assert calculate_recall_at_k(["doc-a"], ["doc-a"], k=0) == 0.0


def test_calculate_reciprocal_rank_uses_first_relevant_position() -> None:
    retrieved = ["doc-a", "doc-b", "doc-c"]
    relevant = ["doc-b"]

    assert calculate_reciprocal_rank(retrieved, relevant) == 0.5


def test_calculate_reciprocal_rank_no_relevant_returns_zero() -> None:
    assert calculate_reciprocal_rank(["doc-a"], ["doc-b"]) == 0.0


def test_calculate_mrr_averages_reciprocal_ranks() -> None:
    results = [
        {"retrieved_docs": ["doc-a", "doc-b"], "relevant_docs": ["doc-b"]},
        {"retrieved_docs": ["doc-x"], "relevant_docs": ["doc-y"]},
    ]

    assert calculate_mrr(results) == 0.25


def test_calculate_mrr_empty_results_returns_zero() -> None:
    assert calculate_mrr([]) == 0.0


def test_calculate_ndcg_at_k_rewards_higher_ranked_relevant_docs() -> None:
    retrieved = ["doc-a", "doc-b", "doc-c"]
    relevant = ["doc-b", "doc-c"]

    assert calculate_ndcg_at_k(retrieved, relevant, k=3) > 0.0
    assert calculate_ndcg_at_k(retrieved, relevant, k=3) < 1.0


def test_calculate_ndcg_at_k_perfect_ranking_returns_one() -> None:
    retrieved = ["doc-a", "doc-b"]
    relevant = ["doc-a", "doc-b"]

    assert calculate_ndcg_at_k(retrieved, relevant, k=2) == 1.0


def test_calculate_ndcg_at_k_no_relevant_returns_zero() -> None:
    assert calculate_ndcg_at_k(["doc-a"], ["doc-b"], k=3) == 0.0


def test_calculate_ndcg_at_k_zero_k_returns_zero() -> None:
    assert calculate_ndcg_at_k(["doc-a"], ["doc-a"], k=0) == 0.0


def test_calculate_answer_contains_score_returns_none_without_expectations() -> None:
    assert calculate_answer_contains_score("answer", None) is None


def test_calculate_citation_compliance_checks_cited_answer() -> None:
    documents = [_page_doc(chunk_id="chunk-1")]
    context = build_citation_context(documents)
    cited_answer = build_cited_answer_from_claim_specs(
        [("Claim with citation.", ["chunk-1"])],
        context,
    )

    assert calculate_citation_compliance(cited_answer, should_cite=True) is True
