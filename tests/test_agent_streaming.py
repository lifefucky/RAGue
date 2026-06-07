from __future__ import annotations

from langchain_core.documents import Document

from rague.agents.workflows import (
    AgentWorkflowConfig,
    GeneratedAnswer,
    RelevanceDecision,
    stream_agentic_rag,
)


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


def test_stream_agentic_rag_emits_expected_events() -> None:
    events = list(
        stream_agentic_rag(
            "Что такое LangGraph?",
            retriever=lambda query: [_page_doc(chunk_id="chunk-1", text=query)],
            should_retrieve=lambda question: True,
            grade_documents=lambda query, docs: RelevanceDecision(is_relevant=True),
            generate_answer=lambda question, docs, context: GeneratedAnswer(
                answer_text=f"Answer for {question}"
            ),
            rewrite_query=lambda question, query, docs: query,
            config=AgentWorkflowConfig(max_rewrites=1),
        )
    )

    event_types = [event.event_type for event in events]
    assert event_types == [
        "agent_decision",
        "retrieval_finished",
        "documents_graded",
        "answer_generated",
        "workflow_finished",
    ]
    assert events[-1].data["answer"]


def test_stream_agentic_rag_rewrite_path_includes_query_rewritten() -> None:
    events = list(
        stream_agentic_rag(
            "Что такое LangGraph?",
            retriever=lambda query: [_page_doc(chunk_id="chunk-1", text=query)],
            should_retrieve=lambda question: True,
            grade_documents=lambda query, docs: RelevanceDecision(is_relevant=False),
            generate_answer=lambda question, docs, context: GeneratedAnswer(
                answer_text="Fallback"
            ),
            rewrite_query=lambda question, query, docs: f"{query} refined",
            config=AgentWorkflowConfig(max_rewrites=1),
        )
    )

    event_types = [event.event_type for event in events]
    assert "query_rewritten" in event_types
    assert event_types[-1] == "workflow_finished"
