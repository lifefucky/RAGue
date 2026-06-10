"""Routing evaluation for should_retrieve decisions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rague.evaluation.dataset import EvaluationCase


def evaluate_should_retrieve_cases(
    cases: Sequence[EvaluationCase],
    decide_should_retrieve: Callable[[str], bool],
) -> dict[str, object]:
    """Evaluate whether routing decisions match labeled should_retrieve flags."""
    mismatches: list[dict[str, object]] = []
    correct = 0

    for case in cases:
        actual = bool(decide_should_retrieve(case.question))
        expected = case.should_retrieve
        if actual == expected:
            correct += 1
        else:
            mismatches.append(
                {
                    "case_id": case.id,
                    "question": case.question,
                    "expected": expected,
                    "actual": actual,
                }
            )

    case_count = len(cases)
    accuracy = correct / case_count if case_count else 0.0
    return {
        "case_count": case_count,
        "correct": correct,
        "accuracy": accuracy,
        "mismatches": mismatches,
    }
