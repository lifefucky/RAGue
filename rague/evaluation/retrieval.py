"""Retrieval evaluation adapters for LangChain retrievers and documents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from rague.evaluation.dataset import EvaluationCase
from rague.evaluation.runner import evaluate_retrieval_cases


def document_id_for_evaluation(
    document: Document,
    *,
    id_field: str = "chunk_id",
) -> str:
    """Extract a stable evaluation id from a retrieved document."""
    metadata = document.metadata or {}
    value = metadata.get(id_field)
    if value is not None and str(value).strip():
        return str(value)
    if document.id is not None and str(document.id).strip():
        return str(document.id)
    raise ValueError(f"Document is missing evaluation id field: {id_field}")


def documents_to_evaluation_ids(
    documents: list[Document],
    *,
    id_field: str = "chunk_id",
) -> list[str]:
    """Convert retrieved documents into ordered evaluation identifiers."""
    return [
        document_id_for_evaluation(document, id_field=id_field)
        for document in documents
    ]


def retriever_to_retrieve_ids(
    retriever: Any,
    *,
    id_field: str = "chunk_id",
) -> Callable[[str], list[str]]:
    """Wrap a LangChain retriever as a question -> doc-id callable."""

    def retrieve_ids(query: str) -> list[str]:
        documents = retriever.invoke(query)
        if not isinstance(documents, list):
            raise TypeError("Retriever must return a list of documents")
        return documents_to_evaluation_ids(documents, id_field=id_field)

    return retrieve_ids


def evaluate_retriever_cases(
    cases: list[EvaluationCase],
    retriever: Any,
    *,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, object]:
    """Evaluate retrieval cases using per-case relevant_id_field settings."""
    per_case_results: list[dict[str, object]] = []
    precision_sums: dict[int, float] = {k: 0.0 for k in k_values}
    recall_sums: dict[int, float] = {k: 0.0 for k in k_values}
    ndcg_sums: dict[int, float] = {k: 0.0 for k in k_values}

    for case in cases:
        retrieve_ids = retriever_to_retrieve_ids(
            retriever,
            id_field=case.relevant_id_field,
        )
        case_result = evaluate_retrieval_cases(
            [case],
            retrieve_ids,
            k_values=k_values,
        )
        per_case = case_result["per_case"]
        assert isinstance(per_case, list) and len(per_case) == 1
        per_case_results.append(per_case[0])

        precision_at_k = per_case[0]["precision_at_k"]
        recall_at_k = per_case[0]["recall_at_k"]
        ndcg_at_k = per_case[0]["ndcg_at_k"]
        assert isinstance(precision_at_k, dict)
        assert isinstance(recall_at_k, dict)
        assert isinstance(ndcg_at_k, dict)
        for k in k_values:
            precision_sums[k] += float(precision_at_k[k])
            recall_sums[k] += float(recall_at_k[k])
            ndcg_sums[k] += float(ndcg_at_k[k])

    case_count = len(cases)
    return {
        "case_count": case_count,
        "per_case": per_case_results,
        "precision_at_k": {
            k: (precision_sums[k] / case_count if case_count else 0.0)
            for k in k_values
        },
        "recall_at_k": {
            k: (recall_sums[k] / case_count if case_count else 0.0)
            for k in k_values
        },
        "mrr": _aggregate_mrr(per_case_results),
        "ndcg_at_k": {
            k: (ndcg_sums[k] / case_count if case_count else 0.0)
            for k in k_values
        },
    }


def _aggregate_mrr(per_case_results: list[dict[str, object]]) -> float:
    from rague.evaluation.metrics import calculate_mrr

    return calculate_mrr(per_case_results)
