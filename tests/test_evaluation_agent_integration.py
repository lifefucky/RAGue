from __future__ import annotations

import os
from pathlib import Path

import pytest

from rague.evaluation.agent import run_agent_evaluation_cases
from rague.evaluation.dataset import load_evaluation_cases

pytestmark_agent = pytest.mark.skipif(
    os.getenv("RAGUE_RUN_AGENT_INTEGRATION") != "1",
    reason="Set RAGUE_RUN_AGENT_INTEGRATION=1 to run live agent evaluation.",
)

pytestmark_qdrant = pytest.mark.skipif(
    os.getenv("RAGUE_RUN_QDRANT_INTEGRATION") != "1",
    reason="Set RAGUE_RUN_QDRANT_INTEGRATION=1 to run live Qdrant evaluation.",
)

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "evaluation" / "basic_cases.json"
)


def _require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for live agent evaluation.")


@pytestmark_agent
@pytestmark_qdrant
def test_agent_evaluation_smoke_on_live_services() -> None:
    _require_openai_key()
    from rague.agents.workflows import run_agentic_rag_from_env

    cases = load_evaluation_cases(DATASET_PATH)
    selected_ids = {"greeting", "dq-uuid-sql"}
    limited_cases = [case for case in cases if case.id in selected_ids]

    assert len(limited_cases) == 2
    assert {case.id for case in limited_cases} == selected_ids

    results = run_agent_evaluation_cases(limited_cases, run_agentic_rag_from_env)

    assert results["case_count"] == 2
    assert "routing" in results
    assert "generation" in results
