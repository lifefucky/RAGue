"""Opt-in HNSW benchmark harness for vector retrieval."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rague.evaluation.dataset import EvaluationCase
from rague.evaluation.metrics import calculate_recall_at_k


@dataclass(frozen=True)
class HnswBenchmarkConfig:
    """Configuration for one HNSW benchmark run."""

    hnsw_ef: int
    full_scan_threshold: int | None = None
    top_k: int = 5
    query_limit: int = 10
    id_field: str = "chunk_id"
    baseline_hnsw_ef: int = 512


@dataclass
class HnswBenchmarkResult:
    """Measured recall and latency for one HNSW configuration."""

    hnsw_ef: int
    recall_at_k: float
    latency_p50_ms: float
    latency_p95_ms: float
    query_count: int
    errors: list[str] = field(default_factory=list)
    index_time_seconds: float | None = None
    memory_notes: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "hnsw_ef": self.hnsw_ef,
            "recall_at_k": self.recall_at_k,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "query_count": self.query_count,
            "errors": list(self.errors),
            "index_time_seconds": self.index_time_seconds,
            "memory_notes": self.memory_notes,
        }


def run_hnsw_benchmark(
    cases: Sequence[EvaluationCase],
    retrieve_ids_for_ef: Callable[[str, int], list[str]],
    *,
    config: HnswBenchmarkConfig,
) -> dict[str, object]:
    """Compare recall@k and latency against a high-ef baseline."""
    retrieval_cases = [case for case in cases if case.should_retrieve and case.relevant_docs]
    if not retrieval_cases:
        return {
            "configs": [],
            "errors": ["No retrieval cases with relevant_docs available for benchmark."],
        }

    limited_cases = retrieval_cases[: config.query_limit]
    baseline_ids_by_question: dict[str, list[str]] = {}
    benchmark_ids_by_question: dict[str, list[str]] = {}
    baseline_latencies_ms: list[float] = []
    benchmark_latencies_ms: list[float] = []
    errors: list[str] = []

    for case in limited_cases:
        try:
            start = time.perf_counter()
            baseline_ids = retrieve_ids_for_ef(case.question, config.baseline_hnsw_ef)
            baseline_latencies_ms.append((time.perf_counter() - start) * 1000)
            baseline_ids_by_question[case.question] = baseline_ids

            start = time.perf_counter()
            benchmark_ids = retrieve_ids_for_ef(case.question, config.hnsw_ef)
            benchmark_latencies_ms.append((time.perf_counter() - start) * 1000)
            benchmark_ids_by_question[case.question] = benchmark_ids
        except Exception as exc:  # noqa: BLE001 - benchmark should capture runtime failures
            errors.append(f"{case.id}: {exc}")

    recall_scores: list[float] = []
    for case in limited_cases:
        if case.question not in benchmark_ids_by_question:
            continue
        baseline_ids = baseline_ids_by_question.get(case.question, [])
        benchmark_ids = benchmark_ids_by_question[case.question]
        if not baseline_ids:
            recall_scores.append(0.0)
            continue

        baseline_top_k = set(baseline_ids[: config.top_k])
        benchmark_top_k = benchmark_ids[: config.top_k]
        matched = sum(1 for doc_id in benchmark_top_k if doc_id in baseline_top_k)
        recall_scores.append(matched / min(len(baseline_top_k), config.top_k))

    result = HnswBenchmarkResult(
        hnsw_ef=config.hnsw_ef,
        recall_at_k=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        latency_p50_ms=_percentile(benchmark_latencies_ms, 50),
        latency_p95_ms=_percentile(benchmark_latencies_ms, 95),
        query_count=len(limited_cases),
        errors=errors,
        memory_notes=(
            "OOM/memory profiling not implemented; inspect errors list for runtime failures."
        ),
    )

    baseline_result = HnswBenchmarkResult(
        hnsw_ef=config.baseline_hnsw_ef,
        recall_at_k=1.0 if recall_scores else 0.0,
        latency_p50_ms=_percentile(baseline_latencies_ms, 50),
        latency_p95_ms=_percentile(baseline_latencies_ms, 95),
        query_count=len(limited_cases),
        errors=errors,
    )

    return {
        "top_k": config.top_k,
        "query_limit": config.query_limit,
        "id_field": config.id_field,
        "configs": [baseline_result.to_dict(), result.to_dict()],
        "errors": errors,
    }


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100)[percentile - 1]


def recall_against_labels(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    *,
    top_k: int,
) -> float:
    """Helper for label-based recall checks in unit tests."""
    return calculate_recall_at_k(retrieved_ids, relevant_ids, top_k)
