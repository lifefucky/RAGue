"""Optional RAGAS-based evaluation wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rague.evaluation.dataset import EvaluationCase
from rague.evaluation.generation import GeneratedEvaluationAnswer

RAGAS_IMPORT_ERROR = (
    "ragas is not installed. Install it separately to run faithfulness and "
    "answer relevance evaluation."
)


def _require_ragas() -> Any:
    try:
        import ragas  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(RAGAS_IMPORT_ERROR) from exc
    return ragas


def evaluate_ragas_cases(
    cases: Sequence[EvaluationCase],
    answers: Sequence[GeneratedEvaluationAnswer],
    *,
    contexts: Sequence[list[str]] | None = None,
) -> dict[str, object]:
    """Evaluate faithfulness and answer relevance with RAGAS when installed.

    This wrapper is intentionally minimal. Live RAGAS scoring is expected to be
    wired in opt-in integration tests once the dependency is approved.
    """
    ragas = _require_ragas()
    if len(cases) != len(answers):
        raise ValueError("cases and answers must have the same length")

    return {
        "case_count": len(cases),
        "ragas_version": getattr(ragas, "__version__", "unknown"),
        "faithfulness": None,
        "answer_relevance": None,
        "status": "ragas_installed_wrapper_ready",
    }
