from __future__ import annotations

import pytest

from rague.evaluation.generation import GeneratedEvaluationAnswer
from rague.evaluation.ragas_eval import RAGAS_IMPORT_ERROR, evaluate_ragas_cases


def test_evaluate_ragas_cases_raises_without_dependency() -> None:
    pytest.importorskip("sys")
    from rague.evaluation.dataset import EvaluationCase

    case = EvaluationCase(
        id="sample",
        question="Что такое LangGraph?",
        expected_answer_contains=["граф"],
        relevant_docs=["doc-1"],
        should_retrieve=True,
        should_cite=True,
    )
    answer = GeneratedEvaluationAnswer(answer_text="LangGraph workflow graph")

    try:
        import ragas  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="ragas is not installed"):
            evaluate_ragas_cases([case], [answer])
    else:
        result = evaluate_ragas_cases([case], [answer])
        assert result["status"] == "ragas_installed_wrapper_ready"
