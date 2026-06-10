"""Trace schema and helpers for agent evaluation diagnostics."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document

from rague.evaluation.dataset import EvaluationCase
from rague.evaluation.metrics import (
    calculate_precision_at_k,
    calculate_reciprocal_rank,
    calculate_recall_at_k,
)
from rague.evaluation.retrieval import document_id_for_evaluation, documents_to_evaluation_ids

_SENSITIVE_ENV_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "CONFLUENCE_PASSWORD",
        "CONFLUENCE_API_TOKEN",
    }
)
_DEFAULT_SNIPPET_CHARS = 200


class AgentDecisionObserver(Protocol):
    """Receive parsed LLM task outputs for evaluation tracing."""

    def on_task_output(
        self,
        task_name: str,
        variables: dict[str, Any],
        output: Any,
    ) -> None: ...


def _truncate_text(value: str, *, max_chars: int = _DEFAULT_SNIPPET_CHARS) -> str:
    stripped = value.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 3] + "..."


def sanitize_task_variables(variables: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of prompt variables without huge document bodies."""
    sanitized: dict[str, Any] = {}
    for key, value in variables.items():
        if key == "documents_context" and isinstance(value, str):
            sanitized[key] = {
                "chars": len(value),
                "preview": _truncate_text(value),
            }
            continue
        if isinstance(value, str):
            sanitized[key] = _truncate_text(value, max_chars=500)
        else:
            sanitized[key] = value
    return sanitized


def summarize_document_row(
    document: Document,
    *,
    rank: int,
    id_field: str,
    relevant_ids: set[str],
    include_snippet: bool = True,
) -> dict[str, Any]:
    """Convert a retrieved document into a compact trace row."""
    metadata = document.metadata or {}
    eval_id = document_id_for_evaluation(document, id_field=id_field)
    row: dict[str, Any] = {
        "rank": rank,
        "eval_id": eval_id,
        "chunk_id": metadata.get("chunk_id"),
        "page_id": metadata.get("page_id"),
        "document_id": metadata.get("document_id"),
        "title": metadata.get("title"),
        "is_relevant": eval_id in relevant_ids,
    }
    for score_field in ("rerank_score", "vector_score", "bm25_score"):
        value = metadata.get(score_field)
        if value is not None:
            row[score_field] = value
    retrieval_sources = metadata.get("retrieval_sources")
    if retrieval_sources:
        row["retrieval_sources"] = list(retrieval_sources)
    if include_snippet and document.page_content:
        row["snippet"] = _truncate_text(document.page_content)
    return row


def summarize_retrieval_attempt(
    *,
    query: str,
    documents: Sequence[Document],
    case: EvaluationCase,
    attempt_index: int,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Build a retrieval funnel record for one query attempt."""
    relevant_ids = set(case.relevant_docs)
    id_field = case.relevant_id_field
    retrieved_ids = documents_to_evaluation_ids(list(documents), id_field=id_field)
    rows = [
        summarize_document_row(
            document,
            rank=index + 1,
            id_field=id_field,
            relevant_ids=relevant_ids,
        )
        for index, document in enumerate(documents)
    ]

    metrics: dict[str, Any] = {
        "reciprocal_rank": calculate_reciprocal_rank(retrieved_ids, case.relevant_docs),
    }
    for k in k_values:
        metrics[f"precision_at_{k}"] = calculate_precision_at_k(
            retrieved_ids,
            case.relevant_docs,
            k,
        )
        metrics[f"recall_at_{k}"] = calculate_recall_at_k(
            retrieved_ids,
            case.relevant_docs,
            k,
        )

    relevant_found = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
    return {
        "attempt_index": attempt_index,
        "query": query,
        "document_count": len(documents),
        "retrieved_ids": retrieved_ids,
        "relevant_ids": list(case.relevant_docs),
        "relevant_found_in_results": relevant_found,
        "documents": rows,
        "metrics": metrics,
    }


def case_context(case: EvaluationCase) -> dict[str, Any]:
    """Serialize labeled case metadata for trace records."""
    return {
        "case_id": case.id,
        "question": case.question,
        "expected_answer_contains": case.expected_answer_contains,
        "relevant_docs": list(case.relevant_docs),
        "relevant_id_field": case.relevant_id_field,
        "should_retrieve": case.should_retrieve,
        "should_cite": case.should_cite,
        "query_type": case.query_type,
        "notes": case.notes,
    }


def safe_run_metadata() -> dict[str, Any]:
    """Capture non-secret runtime configuration for trace headers."""
    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "qdrant_collection": os.getenv("QDRANT_COLLECTION"),
        "qdrant_url": os.getenv("QDRANT_URL"),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "embedding_vector_size": os.getenv("EMBEDDING_VECTOR_SIZE"),
        "reranker_model": os.getenv("RERANKER_MODEL"),
        "retrieval_top_k": os.getenv("RETRIEVAL_TOP_K"),
        "retrieval_bm25_limit": os.getenv("RETRIEVAL_BM25_LIMIT"),
        "retrieval_vector_limit": os.getenv("RETRIEVAL_VECTOR_LIMIT"),
        "chat_model": os.getenv("RAGUE_CHAT_MODEL", "gpt-4o-mini"),
        "max_rewrites": os.getenv("RAGUE_MAX_REWRITES"),
    }
    for secret_key in _SENSITIVE_ENV_KEYS:
        if secret_key in metadata:
            metadata.pop(secret_key, None)
    return metadata


@dataclass
class TraceRecorder:
    """Accumulate per-case agent and retrieval trace steps."""

    case: EvaluationCase
    steps: list[dict[str, Any]] = field(default_factory=list)
    retrieval_attempts: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float | None = None

    def on_task_output(
        self,
        task_name: str,
        variables: dict[str, Any],
        output: Any,
    ) -> None:
        """Observer callback for parsed LLM task outputs."""
        payload: dict[str, Any] = {
            "task": task_name,
            "variables": sanitize_task_variables(variables),
        }

        if task_name == "should_retrieve":
            payload.update(
                {
                    "needs_retrieval": bool(getattr(output, "needs_retrieval", False)),
                    "reason": getattr(output, "reason", "") or None,
                    "routing_correct": bool(getattr(output, "needs_retrieval", False))
                    == self.case.should_retrieve,
                }
            )
            self.steps.append({"type": "routing", **payload})
            return

        if task_name == "grade_documents":
            payload.update(
                {
                    "is_relevant": bool(getattr(output, "is_relevant", False)),
                    "reason": getattr(output, "reason", "") or None,
                }
            )
            self.steps.append({"type": "grade_documents", **payload})
            return

        if task_name == "rewrite_query":
            payload.update(
                {
                    "rewritten_query": getattr(output, "query", "") or None,
                    "reason": getattr(output, "reason", "") or None,
                }
            )
            self.steps.append({"type": "rewrite_query", **payload})
            return

        if task_name == "generate_answer":
            claims = [
                {
                    "text": claim.text,
                    "chunk_ids": list(claim.chunk_ids),
                }
                for claim in getattr(output, "claims", []) or []
            ]
            payload.update(
                {
                    "answer_text": getattr(output, "answer_text", None),
                    "intro": getattr(output, "intro", None),
                    "summary": getattr(output, "summary", None),
                    "claims": claims,
                }
            )
            self.steps.append({"type": "generate_answer", **payload})

    def record_retrieval(self, query: str, documents: Sequence[Document]) -> None:
        """Record one retrieval funnel attempt."""
        attempt = summarize_retrieval_attempt(
            query=query,
            documents=documents,
            case=self.case,
            attempt_index=len(self.retrieval_attempts),
        )
        self.retrieval_attempts.append(attempt)
        self.steps.append({"type": "retrieval", **attempt})

    def record_final_state(self, state: dict[str, Any]) -> None:
        """Attach workflow final state summary."""
        relevance = state.get("relevance_decision")
        self.steps.append(
            {
                "type": "workflow_finished",
                "should_retrieve": state.get("should_retrieve"),
                "final_query": state.get("query"),
                "retry_count": state.get("retry_count", 0),
                "relevance_decision": {
                    "is_relevant": getattr(relevance, "is_relevant", None),
                    "reason": getattr(relevance, "reason", None),
                }
                if relevance is not None
                else None,
                "answer_preview": _truncate_text(str(state.get("answer", "")), max_chars=500),
                "has_cited_answer": state.get("cited_answer") is not None,
            }
        )

    def record_metrics(self, metrics: dict[str, Any]) -> None:
        """Attach per-case evaluation metrics."""
        self.steps.append({"type": "metrics", **metrics})

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full per-case trace record."""
        return {
            "case": case_context(self.case),
            "retrieval_attempts": self.retrieval_attempts,
            "steps": self.steps,
            "duration_ms": self.duration_ms,
        }


def make_json_serializable(value: Any) -> Any:
    """Convert trace payloads with dataclasses into JSON-safe structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return make_json_serializable(asdict(value))
    return str(value)


def default_trace_output_path(*, runs_dir: Path | None = None) -> Path:
    """Build a timestamped JSONL path under data/evaluation/runs."""
    root = runs_dir or Path("data/evaluation/runs")
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return root / f"{stamp}_agent_trace.jsonl"


def write_trace_jsonl(path: Path, traces: Sequence[dict[str, Any]]) -> None:
    """Write human-readable trace records with indent=2, separated by blank lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = [
        json.dumps(make_json_serializable(trace), ensure_ascii=False, indent=2)
        for trace in traces
    ]
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def read_trace_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load pretty-printed or compact JSONL trace records from a file."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if raw.startswith("["):
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError("Trace JSON array must contain a list of records.")
        return payload

    pretty_blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]
    if len(pretty_blocks) > 1:
        return [json.loads(block) for block in pretty_blocks]

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) > 1 and all(line.startswith("{") for line in lines):
        try:
            return [json.loads(line) for line in lines]
        except json.JSONDecodeError:
            pass

    if pretty_blocks:
        return [json.loads(pretty_blocks[0])]

    return []


def render_trace_summary_markdown(
    *,
    traces: Sequence[dict[str, Any]],
    run_metadata: dict[str, Any],
    aggregate: dict[str, Any] | None = None,
) -> str:
    """Render a compact human-readable trace summary."""
    lines = [
        "# Agent Trace Summary",
        "",
        "## Run Metadata",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for key, value in sorted(run_metadata.items()):
        lines.append(f"| {key} | {value if value is not None else '—'} |")
    lines.extend(["", "## Cases", ""])

    for trace in traces:
        case = trace.get("case", {})
        case_id = case.get("case_id", "unknown")
        lines.append(f"### {case_id}")
        lines.append("")
        lines.append(f"- Question: {case.get('question', '—')}")
        metrics_step = next(
            (step for step in trace.get("steps", []) if step.get("type") == "metrics"),
            {},
        )
        lines.append(
            f"- Answer contains: {metrics_step.get('contains_score', '—')}; "
            f"citation compliant: {metrics_step.get('citation_compliant', '—')}"
        )

        retrieval_attempts = trace.get("retrieval_attempts", [])
        if retrieval_attempts:
            last_attempt = retrieval_attempts[-1]
            metrics = last_attempt.get("metrics", {})
            lines.append(
                f"- Retrieval recall@10: {metrics.get('recall_at_10', '—')}; "
                f"relevant found: {last_attempt.get('relevant_found_in_results', [])}"
            )

        routing_steps = [
            step for step in trace.get("steps", []) if step.get("type") == "routing"
        ]
        if routing_steps:
            routing = routing_steps[-1]
            lines.append(
                f"- Routing: needs_retrieval={routing.get('needs_retrieval')} "
                f"(reason: {routing.get('reason', '—')})"
            )

        grade_steps = [
            step
            for step in trace.get("steps", [])
            if step.get("type") == "grade_documents"
        ]
        if grade_steps:
            grade = grade_steps[-1]
            lines.append(
                f"- Grade: is_relevant={grade.get('is_relevant')} "
                f"(reason: {grade.get('reason', '—')})"
            )

        lines.append("")

    if aggregate:
        lines.extend(["## Aggregate", ""])
        routing = aggregate.get("routing", {})
        generation = aggregate.get("generation", {})
        lines.append(
            f"- Routing accuracy: {routing.get('accuracy', '—')} "
            f"({routing.get('correct', '—')}/{routing.get('case_count', '—')})"
        )
        lines.append(
            f"- Answer contains accuracy: {generation.get('answer_contains_accuracy', '—')}"
        )
        lines.append(
            f"- Citation compliance: {generation.get('citation_compliance_rate', '—')}"
        )
        lines.append("")

    return "\n".join(lines).strip() + "\n"
