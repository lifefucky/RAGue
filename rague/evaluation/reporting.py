"""Markdown reporting helpers for evaluation runs."""

from __future__ import annotations

from typing import Any


def _format_metric_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Metric | Value |", "| --- | --- |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def render_evaluation_summary_markdown(results: dict[str, Any]) -> str:
    """Render a compact Markdown summary for an evaluation run."""
    sections: list[str] = ["# Evaluation Summary", ""]

    if "retrieval" in results:
        retrieval = results["retrieval"]
        sections.append("## Retrieval")
        rows = [
            ("Cases", str(retrieval.get("case_count", "—"))),
            ("MRR", _format_float(retrieval.get("mrr"))),
        ]
        precision_at_k = retrieval.get("precision_at_k", {})
        recall_at_k = retrieval.get("recall_at_k", {})
        if isinstance(precision_at_k, dict):
            for key, value in sorted(precision_at_k.items()):
                rows.append((f"Precision@{key}", _format_float(value)))
        if isinstance(recall_at_k, dict):
            for key, value in sorted(recall_at_k.items()):
                rows.append((f"Recall@{key}", _format_float(value)))
        sections.append(_format_metric_table(rows))
        sections.append("")

    if "routing" in results:
        routing = results["routing"]
        sections.append("## Routing")
        sections.append(
            _format_metric_table(
                [
                    ("Cases", str(routing.get("case_count", "—"))),
                    ("Accuracy", _format_float(routing.get("accuracy"))),
                    ("Mismatches", str(len(routing.get("mismatches", [])))),
                ]
            )
        )
        sections.append("")

    if "generation" in results:
        generation = results["generation"]
        sections.append("## Generation")
        sections.append(
            _format_metric_table(
                [
                    ("Cases", str(generation.get("case_count", "—"))),
                    (
                        "Answer Contains Accuracy",
                        _format_optional_float(generation.get("answer_contains_accuracy")),
                    ),
                    (
                        "Citation Compliance Rate",
                        _format_float(generation.get("citation_compliance_rate")),
                    ),
                    (
                        "Average Citation Rate",
                        _format_optional_float(generation.get("average_citation_rate")),
                    ),
                ]
            )
        )
        sections.append("")

    if "hnsw_benchmark" in results:
        benchmark = results["hnsw_benchmark"]
        sections.append("## HNSW Benchmark")
        configs = benchmark.get("configs", [])
        if isinstance(configs, list) and configs:
            rows = [
                ("Top K", str(benchmark.get("top_k", "—"))),
                ("Query Limit", str(benchmark.get("query_limit", "—"))),
            ]
            for config in configs:
                if not isinstance(config, dict):
                    continue
                label = f"hnsw_ef={config.get('hnsw_ef', '—')}"
                rows.append((f"{label} recall@{benchmark.get('top_k', 'k')}", _format_float(config.get("recall_at_k"))))
                rows.append((f"{label} latency p50 ms", _format_float(config.get("latency_p50_ms"))))
                rows.append((f"{label} latency p95 ms", _format_float(config.get("latency_p95_ms"))))
            sections.append(_format_metric_table(rows))
        else:
            sections.append("No benchmark configs recorded.")
        sections.append("")

    if "ragas" in results:
        ragas = results["ragas"]
        sections.append("## RAGAS")
        sections.append(
            _format_metric_table(
                [
                    ("Status", str(ragas.get("status", "not measured"))),
                    ("Faithfulness", _format_optional_float(ragas.get("faithfulness"))),
                    (
                        "Answer Relevance",
                        _format_optional_float(ragas.get("answer_relevance")),
                    ),
                ]
            )
        )
        sections.append("")

    return "\n".join(sections).strip() + "\n"


def _format_float(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "not measured"
    return _format_float(value)
