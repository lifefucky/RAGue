from __future__ import annotations

from langchain_core.documents import Document

from rague.evaluation.dataset import EvaluationCase
from rague.evaluation.retrieval import (
    document_id_for_evaluation,
    evaluate_retriever_cases,
    retriever_to_retrieve_ids,
)


def _page_doc(*, chunk_id: str, page_id: str = "131304166") -> Document:
    return Document(
        page_content="sample",
        metadata={
            "chunk_id": chunk_id,
            "page_id": page_id,
            "document_id": "confluence:page:131304166",
        },
        id=chunk_id,
    )


class FakeRetriever:
    def invoke(self, query: str) -> list[Document]:
        if "Debezium" in query:
            return [
                _page_doc(chunk_id="chunk-1"),
                _page_doc(chunk_id="chunk-2"),
            ]
        return [_page_doc(chunk_id="chunk-3")]


def test_document_id_for_evaluation_reads_metadata_field() -> None:
    document = _page_doc(chunk_id="chunk-42")

    assert document_id_for_evaluation(document, id_field="chunk_id") == "chunk-42"
    assert document_id_for_evaluation(document, id_field="page_id") == "131304166"


def test_retriever_to_retrieve_ids_wraps_retriever() -> None:
    retrieve_ids = retriever_to_retrieve_ids(FakeRetriever(), id_field="chunk_id")

    assert retrieve_ids("Как настроить Debezium connector?") == ["chunk-1", "chunk-2"]


def test_evaluate_retriever_cases_uses_per_case_id_field() -> None:
    cases = [
        EvaluationCase(
            id="page-level",
            question="Как настроить Debezium connector?",
            expected_answer_contains=["Debezium"],
            relevant_docs=["131304166"],
            should_retrieve=True,
            should_cite=True,
            relevant_id_field="page_id",
        ),
        EvaluationCase(
            id="chunk-level",
            question="Как настроить Debezium connector?",
            expected_answer_contains=["Debezium"],
            relevant_docs=["chunk-1"],
            should_retrieve=True,
            should_cite=True,
            relevant_id_field="chunk_id",
        ),
    ]

    results = evaluate_retriever_cases(cases, FakeRetriever(), k_values=(1, 2))

    assert results["case_count"] == 2
    assert results["precision_at_k"][1] == 1.0
    assert results["recall_at_k"][1] == 1.0
