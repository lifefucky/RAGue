from __future__ import annotations

from pathlib import Path

from rague.evaluation.dataset import load_evaluation_cases
from rague.evaluation.routing import evaluate_should_retrieve_cases

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "evaluation" / "basic_cases.json"


def test_evaluate_should_retrieve_cases_reports_accuracy() -> None:
    cases = load_evaluation_cases(FIXTURE_PATH)

    def decide(question: str) -> bool:
        return not question.startswith("Привет")

    results = evaluate_should_retrieve_cases(cases, decide)

    assert results["case_count"] == 15
    assert results["accuracy"] == 1.0
    assert results["mismatches"] == []


def test_evaluate_should_retrieve_cases_records_mismatches() -> None:
    cases = load_evaluation_cases(FIXTURE_PATH)

    results = evaluate_should_retrieve_cases(cases, lambda _question: True)

    assert results["accuracy"] < 1.0
    assert any(mismatch["case_id"] == "greeting" for mismatch in results["mismatches"])
