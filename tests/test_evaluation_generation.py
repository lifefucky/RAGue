from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from rague.citations import build_citation_context, build_cited_answer_from_claim_specs
from rague.evaluation.dataset import EvaluationCase, load_evaluation_cases
from rague.evaluation.generation import GeneratedEvaluationAnswer, evaluate_generation_cases
from rague.evaluation.metrics import (
    calculate_answer_contains_score,
    calculate_citation_compliance,
)


def _page_doc(*, chunk_id: str, text: str = "sample") -> Document:
    return Document(
        page_content=text,
        metadata={
            "source_type": "confluence",
            "document_type": "page",
            "document_id": "confluence:page:131304575",
            "chunk_id": chunk_id,
            "page_id": "131304575",
            "title": "Asmodeus DQ",
            "path": "Data/DQ",
            "source": "https://wiki.example/pages/viewpage.action?pageId=131304575",
            "source_updated_at": "2026-06-06T10:00:00+00:00",
            "ingested_at": "2026-06-06T11:00:00+00:00",
        },
        id=chunk_id,
    )


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "evaluation" / "basic_cases.json"


def test_calculate_answer_contains_score_is_case_insensitive() -> None:
    assert calculate_answer_contains_score(
        "SQL mask for guid and uuid in ods.t_010_or_organization_common",
        ["guid", "uuid"],
    ) == 1.0
    assert calculate_answer_contains_score(
        "SQL mask for guid and uuid in ods.t_010_or_organization_common",
        ["missing"],
    ) == 0.0
    assert calculate_answer_contains_score("Hello", None) is None


def test_calculate_citation_compliance_respects_should_cite_flag() -> None:
    documents = [_page_doc(chunk_id="chunk-1")]
    context = build_citation_context(documents)
    cited_answer = build_cited_answer_from_claim_specs(
        [("Claim with citation.", ["chunk-1"])],
        context,
    )

    assert calculate_citation_compliance(None, should_cite=False) is True
    assert calculate_citation_compliance(cited_answer, should_cite=True) is True
    assert calculate_citation_compliance(None, should_cite=True) is False


def test_evaluate_generation_cases_scores_sample_fixture_cases() -> None:
    cases = load_evaluation_cases(FIXTURE_PATH)
    sample_cases = [
        next(case for case in cases if case.id == "dq-uuid-sql"),
        next(case for case in cases if case.id == "greeting"),
    ]

    def answer_fn(question: str) -> GeneratedEvaluationAnswer:
        if question.startswith("Привет"):
            return GeneratedEvaluationAnswer(answer_text="Привет!")
        documents = [
            _page_doc(
                chunk_id="chunk-uuid",
                text="SQL guid uuid ods.t_010_or_organization_common",
            )
        ]
        context = build_citation_context(documents)
        cited_answer = build_cited_answer_from_claim_specs(
            [
                (
                    "Для проверки guid и uuid используется SQL-маска в "
                    "ods.t_010_or_organization_common.",
                    ["chunk-uuid"],
                )
            ],
            context,
        )
        return GeneratedEvaluationAnswer(
            answer_text=(
                "Для проверки guid и uuid используется SQL-маска в "
                "ods.t_010_or_organization_common."
            ),
            cited_answer=cited_answer,
        )

    results = evaluate_generation_cases(sample_cases, answer_fn)

    assert results["case_count"] == 2
    assert results["citation_compliance_rate"] == 1.0
    assert results["answer_contains_accuracy"] == 1.0


def test_evaluate_generation_cases_handles_no_expectations() -> None:
    greeting_case = EvaluationCase(
        id="greeting",
        question="Привет!",
        expected_answer_contains=None,
        relevant_docs=[],
        should_retrieve=False,
        should_cite=False,
    )

    results = evaluate_generation_cases(
        [greeting_case],
        lambda _question: GeneratedEvaluationAnswer(answer_text="Привет!"),
    )

    assert results["answer_contains_accuracy"] is None
    assert results["citation_compliance_rate"] == 1.0
