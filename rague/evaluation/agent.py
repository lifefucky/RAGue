"""Agent end-to-end evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rague.evaluation.dataset import EvaluationCase
from rague.evaluation.generation import (
    GeneratedEvaluationAnswer,
    evaluate_generation_cases,
)
from rague.evaluation.routing import evaluate_should_retrieve_cases


def agent_state_to_generation_answer(state: Mapping[str, Any]) -> GeneratedEvaluationAnswer:
    """Convert agent workflow state into a generation evaluation answer."""
    return GeneratedEvaluationAnswer(
        answer_text=str(state.get("answer", "")),
        cited_answer=state.get("cited_answer"),
    )


def run_agent_evaluation_cases(
    cases: Sequence[EvaluationCase],
    run_agent: Callable[[str], Mapping[str, Any]],
) -> dict[str, object]:
    """Evaluate routing and generation behavior for agent workflow runs."""
    states_by_question = {case.question: run_agent(case.question) for case in cases}

    routing_results = evaluate_should_retrieve_cases(
        cases,
        lambda question: bool(states_by_question[question].get("should_retrieve")),
    )

    generation_results = evaluate_generation_cases(
        cases,
        lambda question: agent_state_to_generation_answer(states_by_question[question]),
    )

    return {
        "case_count": len(cases),
        "routing": routing_results,
        "generation": generation_results,
    }
