from __future__ import annotations

from pathlib import Path

from rague.evaluation.dataset import load_evaluation_cases
from rague.evaluation.runner import evaluate_retrieval_cases

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "evaluation" / "basic_cases.json"


def _fake_retrieve_ids(question: str) -> list[str]:
    mapping = {
        "Какой SQL-запрос используется для проверки guid и маски uuid в ods.t_010_or_organization_common?": [
            "131304575",
            "131304174",
        ],
        "Какие SQL-скрипты помогают диагностировать проблемы pipe в NiFi и pg_exttable в DDL?": [
            "131304485",
            "131304577",
        ],
        "Привет, как дела?": [],
    }
    return mapping.get(question, ["131304241"])


def test_evaluate_retrieval_cases_aggregates_metrics_deterministically() -> None:
    cases = load_evaluation_cases(FIXTURE_PATH)
    retrieval_cases = [
        next(case for case in cases if case.id == "dq-uuid-sql"),
        next(case for case in cases if case.id == "multi-nifi-ddl-pipes"),
    ]

    first_run = evaluate_retrieval_cases(retrieval_cases, _fake_retrieve_ids, k_values=(1, 3))
    second_run = evaluate_retrieval_cases(retrieval_cases, _fake_retrieve_ids, k_values=(1, 3))

    assert first_run == second_run
    assert first_run["case_count"] == 2
    assert first_run["precision_at_k"][1] == 1.0
    assert first_run["mrr"] == 1.0


def test_evaluate_retrieval_cases_handles_empty_relevant_docs() -> None:
    cases = load_evaluation_cases(FIXTURE_PATH)
    greeting_case = next(case for case in cases if case.id == "greeting")

    result = evaluate_retrieval_cases([greeting_case], _fake_retrieve_ids, k_values=(1,))

    per_case = result["per_case"][0]
    assert per_case["reciprocal_rank"] == 0.0
    assert per_case["precision_at_k"][1] == 0.0
    assert per_case["recall_at_k"][1] == 0.0
    assert per_case["ndcg_at_k"][1] == 0.0
