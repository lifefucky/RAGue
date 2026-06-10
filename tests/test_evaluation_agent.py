from __future__ import annotations

from pathlib import Path

from rague.evaluation.agent import (
    agent_state_to_generation_answer,
    run_agent_evaluation_cases,
)
from rague.evaluation.dataset import load_evaluation_cases

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "evaluation" / "basic_cases.json"


def test_agent_state_to_generation_answer_maps_state_fields() -> None:
    answer = agent_state_to_generation_answer(
        {
            "answer": "Final answer",
            "cited_answer": None,
            "should_retrieve": True,
        }
    )

    assert answer.answer_text == "Final answer"
    assert answer.cited_answer is None


def test_run_agent_evaluation_cases_aggregates_routing_and_generation() -> None:
    cases = load_evaluation_cases(FIXTURE_PATH)
    sample_cases = [
        next(case for case in cases if case.id == "dq-uuid-sql"),
        next(case for case in cases if case.id == "greeting"),
    ]

    def fake_run_agent(question: str) -> dict[str, object]:
        if question.startswith("Привет"):
            return {
                "answer": "Привет!",
                "cited_answer": None,
                "should_retrieve": False,
            }
        return {
            "answer": "SQL guid uuid ods.t_010_or_organization_common",
            "cited_answer": None,
            "should_retrieve": True,
        }

    results = run_agent_evaluation_cases(sample_cases, fake_run_agent)

    assert results["case_count"] == 2
    assert results["routing"]["accuracy"] == 1.0
    assert results["generation"]["case_count"] == 2
