from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rague.agents.workflows import GeneratedAnswer, RelevanceDecision, run_agentic_rag_from_env


pytestmark_agent_integration = pytest.mark.skipif(
    os.getenv("RAGUE_RUN_AGENT_INTEGRATION") != "1",
    reason="Set RAGUE_RUN_AGENT_INTEGRATION=1 to run live agent integration tests.",
)

pytestmark_qdrant_integration = pytest.mark.skipif(
    os.getenv("RAGUE_RUN_QDRANT_INTEGRATION") != "1",
    reason="Set RAGUE_RUN_QDRANT_INTEGRATION=1 to run live Qdrant integration tests.",
)


def _require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for live agent integration tests.")


def _page_doc(*, chunk_id: str, text: str = "sample") -> Document:
    metadata = {
        "source_type": "confluence",
        "document_type": "page",
        "document_id": "confluence:page:131304166",
        "chunk_id": chunk_id,
        "page_id": "131304166",
        "title": "Debezium setup",
        "path": "Data/Debezium",
        "source": "https://wiki.example/pages/viewpage.action?pageId=131304166",
        "source_updated_at": "2026-06-06T10:00:00+00:00",
        "ingested_at": "2026-06-06T11:00:00+00:00",
    }
    return Document(page_content=text, metadata=metadata, id=chunk_id)


@pytestmark_agent_integration
def test_llm_should_retrieve_live() -> None:
    _require_openai_key()
    from rague.agents.decisions import AgentLlmDecisions
    from rague.agents.llm import create_chat_model_from_env

    decisions = AgentLlmDecisions(create_chat_model_from_env())
    result = decisions.decide_should_retrieve("Привет!")
    assert isinstance(result, bool)


@pytestmark_agent_integration
def test_llm_generate_answer_live_with_fixture_docs() -> None:
    _require_openai_key()
    from rague.agents.decisions import AgentLlmDecisions
    from rague.agents.llm import create_chat_model_from_env
    from rague.citations import build_citation_context

    documents = [_page_doc(chunk_id="chunk-1", text="LangGraph workflow graph")]
    context = build_citation_context(documents)
    decisions = AgentLlmDecisions(create_chat_model_from_env())

    generated = decisions.generate_answer(
        "Что такое LangGraph?",
        documents,
        context,
    )

    assert generated.answer_text or generated.claim_specs


@pytestmark_agent_integration
@pytestmark_qdrant_integration
def test_agent_end_to_end_live_qdrant_llm() -> None:
    _require_openai_key()
    state = run_agentic_rag_from_env("Что такое LangGraph?")

    assert state.get("answer")
    if state.get("documents"):
        assert "## Источники" in state["answer"] or state.get("cited_answer") is not None


@pytestmark_agent_integration
@pytestmark_qdrant_integration
def test_agent_streaming_live_qdrant_llm() -> None:
    _require_openai_key()
    from rague.agents.workflows import stream_agentic_rag_from_env

    events = list(stream_agentic_rag_from_env("Что такое LangGraph?"))
    assert events
    assert events[-1].event_type == "workflow_finished"


def test_run_agentic_rag_from_env_builds_dependencies(monkeypatch) -> None:
    fake_bundle = MagicMock()
    fake_bundle.config.max_rewrites = 2
    fake_bundle.app.invoke.return_value = {"answer": "mock answer"}

    monkeypatch.setattr(
        "rague.agents.workflows.build_agentic_rag_from_env",
        lambda: fake_bundle,
    )

    state = run_agentic_rag_from_env("test question")
    assert state["answer"] == "mock answer"
    fake_bundle.app.invoke.assert_called_once()
