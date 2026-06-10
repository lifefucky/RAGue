"""Deterministic evaluation runners for labeled datasets."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rague.evaluation.dataset import EvaluationCase
from rague.evaluation.metrics import (
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)


def evaluate_retrieval_cases(
    cases: Sequence[EvaluationCase],
    retrieve_ids: Callable[[str], list[str]],
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, object]:
    """Evaluate retrieval quality for labeled cases using a doc-id retriever."""
    per_case_results: list[dict[str, object]] = []
    precision_sums = {k: 0.0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}
    ndcg_sums = {k: 0.0 for k in k_values}

    for case in cases:
        retrieved_docs = retrieve_ids(case.question)
        case_result: dict[str, object] = {
            "case_id": case.id,
            "question": case.question,
            "retrieved_docs": retrieved_docs,
            "relevant_docs": case.relevant_docs,
            "reciprocal_rank": calculate_reciprocal_rank(
                retrieved_docs,
                case.relevant_docs,
            ),
        }

        precision_at_k: dict[int, float] = {}
        recall_at_k: dict[int, float] = {}
        ndcg_at_k: dict[int, float] = {}
        for k in k_values:
            precision = calculate_precision_at_k(
                retrieved_docs,
                case.relevant_docs,
                k,
            )
            recall = calculate_recall_at_k(
                retrieved_docs,
                case.relevant_docs,
                k,
            )
            ndcg = calculate_ndcg_at_k(
                retrieved_docs,
                case.relevant_docs,
                k,
            )
            precision_at_k[k] = precision
            recall_at_k[k] = recall
            ndcg_at_k[k] = ndcg
            precision_sums[k] += precision
            recall_sums[k] += recall
            ndcg_sums[k] += ndcg

        case_result["precision_at_k"] = precision_at_k
        case_result["recall_at_k"] = recall_at_k
        case_result["ndcg_at_k"] = ndcg_at_k
        per_case_results.append(case_result)

    case_count = len(cases)
    aggregate_precision = {
        k: (precision_sums[k] / case_count if case_count else 0.0)
        for k in k_values
    }
    aggregate_recall = {
        k: (recall_sums[k] / case_count if case_count else 0.0)
        for k in k_values
    }
    aggregate_ndcg = {
        k: (ndcg_sums[k] / case_count if case_count else 0.0)
        for k in k_values
    }

    return {
        "case_count": case_count,
        "per_case": per_case_results,
        "precision_at_k": aggregate_precision,
        "recall_at_k": aggregate_recall,
        "mrr": calculate_mrr(per_case_results),
        "ndcg_at_k": aggregate_ndcg,
    }
