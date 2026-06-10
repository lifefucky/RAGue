from __future__ import annotations

import os
from pathlib import Path

import pytest

from rague.evaluation.dataset import load_evaluation_cases
from rague.evaluation.retrieval import evaluate_retriever_cases

pytestmark = pytest.mark.skipif(
    os.getenv("RAGUE_RUN_QDRANT_INTEGRATION") != "1",
    reason="Set RAGUE_RUN_QDRANT_INTEGRATION=1 to run Qdrant retrieval evaluation.",
)

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "evaluation" / "basic_cases.json"
)


def test_retrieval_evaluation_against_live_qdrant_corpus() -> None:
    from rague.retrieval.hybrid_reranker import create_retriever_from_env

    cases = load_evaluation_cases(DATASET_PATH)
    retrieval_cases = [case for case in cases if case.should_retrieve]
    retriever = create_retriever_from_env()

    results = evaluate_retriever_cases(retrieval_cases, retriever, k_values=(1, 5))

    assert results["case_count"] == len(retrieval_cases)
    assert isinstance(results["precision_at_k"][1], float)
    assert isinstance(results["recall_at_k"][5], float)
    assert isinstance(results["mrr"], float)
    assert isinstance(results["ndcg_at_k"][5], float)
