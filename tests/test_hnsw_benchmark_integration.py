from __future__ import annotations

import os
from pathlib import Path

import pytest

from rague.evaluation.dataset import load_evaluation_cases
from rague.evaluation.hnsw_benchmark import HnswBenchmarkConfig, run_hnsw_benchmark
from rague.evaluation.retrieval import retriever_to_retrieve_ids

pytestmark = pytest.mark.skipif(
    os.getenv("RAGUE_RUN_HNSW_BENCHMARK") != "1",
    reason="Set RAGUE_RUN_HNSW_BENCHMARK=1 to run HNSW benchmark integration test.",
)

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "evaluation" / "basic_cases.json"
)


def test_hnsw_benchmark_against_live_qdrant() -> None:
    from dataclasses import replace

    from rague.retrieval.hybrid_reranker import create_retriever_from_config, _config_from_env

    cases = load_evaluation_cases(DATASET_PATH)

    def retrieve_ids_for_ef(question: str, hnsw_ef: int) -> list[str]:
        config = replace(_config_from_env(), hnsw_ef_search=hnsw_ef)
        retriever = create_retriever_from_config(config)
        retrieve_ids = retriever_to_retrieve_ids(retriever, id_field="page_id")
        return retrieve_ids(question)

    result = run_hnsw_benchmark(
        cases,
        retrieve_ids_for_ef,
        config=HnswBenchmarkConfig(
            hnsw_ef=64,
            baseline_hnsw_ef=256,
            top_k=5,
            query_limit=3,
            id_field="page_id",
        ),
    )

    assert len(result["configs"]) == 2
    assert result["configs"][1]["query_count"] <= 3
