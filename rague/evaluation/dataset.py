"""Labeled evaluation dataset models and loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_RELEVANT_ID_FIELDS = frozenset({"chunk_id", "document_id", "page_id"})
DEFAULT_RELEVANT_ID_FIELD = "chunk_id"


@dataclass(frozen=True)
class EvaluationCase:
    """Single labeled question for retrieval or generation evaluation."""

    id: str
    question: str
    expected_answer_contains: list[str] | None
    relevant_docs: list[str]
    should_retrieve: bool
    should_cite: bool
    query_type: str = "fact_lookup"
    relevant_id_field: str = DEFAULT_RELEVANT_ID_FIELD
    notes: str | None = None


_REQUIRED_FIELDS = (
    "id",
    "question",
    "relevant_docs",
    "should_retrieve",
    "should_cite",
)


def case_relevant_ids(case: EvaluationCase) -> list[str]:
    """Return labeled relevant document identifiers for a case."""
    return list(case.relevant_docs)


def _parse_case(raw: dict[str, Any], *, index: int) -> EvaluationCase:
    missing = [field for field in _REQUIRED_FIELDS if field not in raw]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Evaluation case at index {index} is missing fields: {joined}")

    expected = raw.get("expected_answer_contains")
    if expected is not None and not isinstance(expected, list):
        raise ValueError(
            f"Evaluation case at index {index} has invalid expected_answer_contains"
        )

    relevant_docs = raw["relevant_docs"]
    if not isinstance(relevant_docs, list):
        raise ValueError(f"Evaluation case at index {index} has invalid relevant_docs")

    relevant_id_field = str(raw.get("relevant_id_field", DEFAULT_RELEVANT_ID_FIELD))
    if relevant_id_field not in VALID_RELEVANT_ID_FIELDS:
        raise ValueError(
            f"Evaluation case at index {index} has invalid relevant_id_field: "
            f"{relevant_id_field}"
        )

    notes = raw.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"Evaluation case at index {index} has invalid notes")

    return EvaluationCase(
        id=str(raw["id"]),
        question=str(raw["question"]),
        expected_answer_contains=expected,
        relevant_docs=[str(doc_id) for doc_id in relevant_docs],
        should_retrieve=bool(raw["should_retrieve"]),
        should_cite=bool(raw["should_cite"]),
        query_type=str(raw.get("query_type", "fact_lookup")),
        relevant_id_field=relevant_id_field,
        notes=notes,
    )


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """Load labeled evaluation cases from a JSON list file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must be a JSON list")

    return [_parse_case(item, index=index) for index, item in enumerate(payload)]
