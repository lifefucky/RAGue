from __future__ import annotations

import os
from pathlib import Path

import pytest

from rague.evaluation.dataset import EvaluationCase, load_evaluation_cases
from rague.evaluation.metrics import calculate_recall_at_k
from rague.evaluation.retrieval import documents_to_evaluation_ids, evaluate_retriever_cases

pytestmark = pytest.mark.skipif(
    os.getenv("RAGUE_RUN_QDRANT_INTEGRATION") != "1",
    reason="Set RAGUE_RUN_QDRANT_INTEGRATION=1 to validate fixture retrieval against Qdrant.",
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "evaluation" / "basic_cases.json"


def test_fixture_cases_retrieve_expected_pages_from_qdrant() -> None:
    from rague.retrieval.hybrid_reranker import create_retriever_from_env

    cases = load_evaluation_cases(FIXTURE_PATH)
    retrieval_cases = [case for case in cases if case.should_retrieve]
    retriever = create_retriever_from_env()

    results = evaluate_retriever_cases(retrieval_cases, retriever, k_values=(10,))
    per_case = results["per_case"]
    assert isinstance(per_case, list)

    failures: list[str] = []
    for case, case_result in zip(retrieval_cases, per_case, strict=True):
        assert isinstance(case, EvaluationCase)
        recall_at_10 = case_result["recall_at_k"][10]
        if recall_at_10 < 1.0:
            retrieved = documents_to_evaluation_ids(
                retriever.invoke(case.question),
                id_field=case.relevant_id_field,
            )[:10]
            failures.append(
                f"{case.id}: expected {case.relevant_docs}, got top10={retrieved}"
            )

    assert not failures, "Fixture retrieval validation failed:\n" + "\n".join(failures)


def test_fixture_greeting_case_has_no_relevant_docs() -> None:
    cases = load_evaluation_cases(FIXTURE_PATH)
    greeting = next(case for case in cases if case.id == "greeting")
    assert greeting.relevant_docs == []
    assert greeting.should_retrieve is False
