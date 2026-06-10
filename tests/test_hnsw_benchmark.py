from __future__ import annotations

from rague.evaluation.dataset import EvaluationCase
from rague.evaluation.hnsw_benchmark import HnswBenchmarkConfig, run_hnsw_benchmark


def test_run_hnsw_benchmark_compares_configs_deterministically() -> None:
    cases = [
        EvaluationCase(
            id="case-1",
            question="Как настроить Debezium connector?",
            expected_answer_contains=["Debezium"],
            relevant_docs=["131304166"],
            should_retrieve=True,
            should_cite=True,
            relevant_id_field="page_id",
        )
    ]

    def retrieve_ids_for_ef(question: str, hnsw_ef: int) -> list[str]:
        if hnsw_ef >= 256:
            return ["131304166", "other-page"]
        return ["other-page", "missing-page"]

    result = run_hnsw_benchmark(
        cases,
        retrieve_ids_for_ef,
        config=HnswBenchmarkConfig(hnsw_ef=64, baseline_hnsw_ef=512, top_k=2),
    )

    configs = result["configs"]
    assert len(configs) == 2
    assert configs[0]["hnsw_ef"] == 512
    assert configs[1]["hnsw_ef"] == 64
    assert configs[1]["recall_at_k"] == 0.5
    assert configs[1]["latency_p50_ms"] >= 0.0


def test_run_hnsw_benchmark_handles_no_retrieval_cases() -> None:
    greeting_case = EvaluationCase(
        id="greeting",
        question="Привет!",
        expected_answer_contains=None,
        relevant_docs=[],
        should_retrieve=False,
        should_cite=False,
    )

    result = run_hnsw_benchmark(
        [greeting_case],
        lambda _question, _ef: [],
        config=HnswBenchmarkConfig(hnsw_ef=64),
    )

    assert result["configs"] == []
    assert result["errors"]
