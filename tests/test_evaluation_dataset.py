from __future__ import annotations

import json
from pathlib import Path

import pytest

from rague.evaluation.dataset import EvaluationCase, load_evaluation_cases

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "evaluation" / "basic_cases.json"


def test_load_evaluation_cases_reads_fixture() -> None:
    cases = load_evaluation_cases(FIXTURE_PATH)

    assert len(cases) == 15
    assert all(isinstance(case, EvaluationCase) for case in cases)


def test_load_evaluation_cases_parses_code_lookup_case() -> None:
    cases = load_evaluation_cases(FIXTURE_PATH)
    dq_case = next(case for case in cases if case.id == "dq-uuid-sql")

    assert dq_case.question.startswith("Какой SQL-запрос")
    assert dq_case.relevant_docs == ["131304575"]
    assert dq_case.relevant_id_field == "page_id"
    assert dq_case.query_type == "code_lookup"
    assert dq_case.should_retrieve is True
    assert dq_case.should_cite is True


def test_load_evaluation_cases_parses_multi_page_case() -> None:
    cases = load_evaluation_cases(FIXTURE_PATH)
    multi_case = next(case for case in cases if case.id == "multi-nifi-ddl-pipes")

    assert multi_case.relevant_docs == ["131304485", "131304577"]
    assert multi_case.query_type == "long_context"


def test_load_evaluation_cases_parses_greeting_case() -> None:
    cases = load_evaluation_cases(FIXTURE_PATH)
    greeting_case = next(case for case in cases if case.id == "greeting")

    assert greeting_case.relevant_docs == []
    assert greeting_case.should_retrieve is False
    assert greeting_case.should_cite is False
    assert greeting_case.expected_answer_contains is None


def test_load_evaluation_cases_applies_default_optional_fields(
    tmp_path: Path,
) -> None:
    payload = [
        {
            "id": "minimal",
            "question": "Minimal case",
            "relevant_docs": ["page-1"],
            "should_retrieve": True,
            "should_cite": True,
        }
    ]
    dataset_path = tmp_path / "minimal_cases.json"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    case = load_evaluation_cases(dataset_path)[0]

    assert case.query_type == "fact_lookup"
    assert case.relevant_id_field == "chunk_id"
    assert case.notes is None


def test_load_evaluation_cases_reads_corpus_bound_dataset() -> None:
    corpus_path = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "basic_cases.json"
    cases = load_evaluation_cases(corpus_path)

    assert len(cases) == 15
    dq_case = next(case for case in cases if case.id == "dq-uuid-sql")
    assert dq_case.relevant_id_field == "page_id"
    assert dq_case.relevant_docs == ["131304575"]
    assert dq_case.query_type == "code_lookup"


def test_load_evaluation_cases_raises_on_missing_required_field(
    tmp_path: Path,
) -> None:
    payload = [
        {
            "id": "broken",
            "question": "Broken case",
            "relevant_docs": [],
            "should_retrieve": False,
        }
    ]
    dataset_path = tmp_path / "broken_cases.json"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields: should_cite"):
        load_evaluation_cases(dataset_path)
