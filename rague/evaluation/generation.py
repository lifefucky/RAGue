"""Lightweight generation evaluation without LLM judges."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rague.citations.models import CitedAnswer
from rague.evaluation.dataset import EvaluationCase
from rague.evaluation.metrics import (
    calculate_answer_contains_score,
    calculate_citation_compliance,
    calculate_citation_rate,
)


@dataclass(frozen=True)
class GeneratedEvaluationAnswer:
    """Agent or generator output for evaluation."""

    answer_text: str
    cited_answer: CitedAnswer | None = None


def evaluate_generation_cases(
    cases: Sequence[EvaluationCase],
    answer_fn: Callable[[str], GeneratedEvaluationAnswer],
) -> dict[str, object]:
    """Evaluate answer correctness and citation compliance for labeled cases."""
    per_case_results: list[dict[str, object]] = []
    contains_scores: list[float] = []
    citation_compliance_scores: list[float] = []
    citation_rates: list[float] = []

    for case in cases:
        generated = answer_fn(case.question)
        contains_score = calculate_answer_contains_score(
            generated.answer_text,
            case.expected_answer_contains,
        )
        citation_compliant = calculate_citation_compliance(
            generated.cited_answer,
            should_cite=case.should_cite,
        )
        citation_rate = (
            calculate_citation_rate(generated.cited_answer)
            if generated.cited_answer is not None
            else 0.0
        )

        case_result: dict[str, object] = {
            "case_id": case.id,
            "question": case.question,
            "answer_text": generated.answer_text,
            "contains_score": contains_score,
            "citation_rate": citation_rate,
            "citation_compliant": citation_compliant,
            "should_cite": case.should_cite,
        }
        per_case_results.append(case_result)

        if contains_score is not None:
            contains_scores.append(contains_score)
        citation_compliance_scores.append(1.0 if citation_compliant else 0.0)
        if case.should_cite:
            citation_rates.append(citation_rate)

    return {
        "case_count": len(cases),
        "per_case": per_case_results,
        "answer_contains_accuracy": (
            sum(contains_scores) / len(contains_scores)
            if contains_scores
            else None
        ),
        "citation_compliance_rate": (
            sum(citation_compliance_scores) / len(citation_compliance_scores)
            if citation_compliance_scores
            else 0.0
        ),
        "average_citation_rate": (
            sum(citation_rates) / len(citation_rates) if citation_rates else None
        ),
    }
