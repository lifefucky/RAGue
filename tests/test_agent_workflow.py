from __future__ import annotations

from langchain_core.documents import Document

from rague.agents.workflows import (
    AgentWorkflowConfig,
    GeneratedAnswer,
    RelevanceDecision,
    build_agentic_rag_workflow,
    create_retrieval_tool,
    render_generated_answer,
    run_agentic_rag,
)
from rague.citations import build_citation_context


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


def test_no_retrieval_question_skips_retriever() -> None:
    retriever_calls: list[str] = []

    def fake_retriever(query: str) -> list[Document]:
        retriever_calls.append(query)
        return []

    def should_retrieve(question: str) -> bool:
        del question
        return False

    def generate_answer(question, documents, citation_context):
        del question, documents, citation_context
        return GeneratedAnswer(answer_text="Привет!")

    state = run_agentic_rag(
        "Привет!",
        retriever=fake_retriever,
        should_retrieve=should_retrieve,
        grade_documents=lambda query, documents: RelevanceDecision(is_relevant=True),
        generate_answer=generate_answer,
        rewrite_query=lambda question, query, documents: query,
    )

    assert retriever_calls == []
    assert state["should_retrieve"] is False
    assert state["answer"] == "Привет!"
    assert state["cited_answer"] is None


def test_retrieval_question_builds_cited_answer() -> None:
    documents = [
        _page_doc(chunk_id="chunk-1", text="Debezium connector setup"),
        _page_doc(chunk_id="chunk-2", text="Kafka topic config"),
    ]
    retriever_calls: list[str] = []

    def fake_retriever(query: str) -> list[Document]:
        retriever_calls.append(query)
        return documents

    def should_retrieve(question: str) -> bool:
        return "LangGraph" in question

    def generate_answer(question, retrieved_documents, citation_context):
        del question, retrieved_documents
        assert citation_context is not None
        return GeneratedAnswer(
            claim_specs=[
                ("LangGraph — это workflow-граф для агентов.", ["chunk-1"]),
                ("Он поддерживает retrieval и routing.", ["chunk-2"]),
            ]
        )

    state = run_agentic_rag(
        "Что такое LangGraph?",
        retriever=fake_retriever,
        should_retrieve=should_retrieve,
        grade_documents=lambda query, docs: RelevanceDecision(is_relevant=True),
        generate_answer=generate_answer,
        rewrite_query=lambda question, query, documents: query,
    )

    assert retriever_calls == ["Что такое LangGraph?"]
    assert state["should_retrieve"] is True
    assert state["cited_answer"] is not None
    assert "## Источники" in state["answer"]
    assert "LangGraph — это workflow-граф для агентов. [1]" in state["answer"]
    assert "Он поддерживает retrieval и routing. [1]" in state["answer"]


def test_irrelevant_documents_trigger_rewrite_until_limit() -> None:
    retriever_calls: list[str] = []
    rewrite_calls: list[str] = []
    grade_calls = 0

    def fake_retriever(query: str) -> list[Document]:
        retriever_calls.append(query)
        return [_page_doc(chunk_id=f"chunk-{len(retriever_calls)}", text=query)]

    def should_retrieve(question: str) -> bool:
        del question
        return True

    def grade_documents(query, docs):
        nonlocal grade_calls
        grade_calls += 1
        del query, docs
        return RelevanceDecision(is_relevant=False, reason="off-topic")

    def rewrite_query(question, query, documents):
        del question, documents
        rewrite_calls.append(query)
        return f"{query} refined"

    state = run_agentic_rag(
        "Объясни архитектуру LangGraph и её компоненты",
        retriever=fake_retriever,
        should_retrieve=should_retrieve,
        grade_documents=grade_documents,
        generate_answer=lambda question, documents, citation_context: GeneratedAnswer(
            answer_text="Best-effort answer after retries."
        ),
        rewrite_query=rewrite_query,
        config=AgentWorkflowConfig(max_rewrites=2),
    )

    assert retriever_calls == [
        "Объясни архитектуру LangGraph и её компоненты",
        "Объясни архитектуру LangGraph и её компоненты refined",
        "Объясни архитектуру LangGraph и её компоненты refined refined",
    ]
    assert rewrite_calls == [
        "Объясни архитектуру LangGraph и её компоненты",
        "Объясни архитектуру LangGraph и её компоненты refined",
    ]
    assert grade_calls == 3
    assert state["retry_count"] == 2
    assert state["answer"] == "Best-effort answer after retries."


def test_exhausted_retries_do_not_loop_forever() -> None:
    retriever_calls: list[str] = []

    def fake_retriever(query: str) -> list[Document]:
        retriever_calls.append(query)
        return [_page_doc(chunk_id="chunk-1")]

    state = run_agentic_rag(
        "Что такое LangGraph?",
        retriever=fake_retriever,
        should_retrieve=lambda question: True,
        grade_documents=lambda query, docs: RelevanceDecision(is_relevant=False),
        generate_answer=lambda question, documents, citation_context: GeneratedAnswer(
            answer_text="Fallback answer."
        ),
        rewrite_query=lambda question, query, documents: f"{query}!",
        config=AgentWorkflowConfig(max_rewrites=1),
    )

    assert len(retriever_calls) == 2
    assert state["retry_count"] == 1
    assert state["answer"] == "Fallback answer."


def test_create_retrieval_tool_wraps_retriever() -> None:
    calls: list[str] = []

    def fake_retriever(query: str) -> list[Document]:
        calls.append(query)
        return [_page_doc(chunk_id="chunk-1", text=query)]

    tool = create_retrieval_tool(fake_retriever)
    result = tool.invoke("LangGraph workflow")

    assert calls == ["LangGraph workflow"]
    assert len(result) == 1
    assert result[0].metadata["chunk_id"] == "chunk-1"


def test_render_generated_answer_plain_text() -> None:
    answer_text, cited_answer = render_generated_answer(
        GeneratedAnswer(answer_text="Plain response."),
        None,
    )

    assert answer_text == "Plain response."
    assert cited_answer is None


def test_render_generated_answer_with_claim_specs() -> None:
    documents = [_page_doc(chunk_id="chunk-1")]
    context = build_citation_context(documents)

    answer_text, cited_answer = render_generated_answer(
        GeneratedAnswer(claim_specs=[("Claim with citation.", ["chunk-1"])]),
        context,
    )

    assert cited_answer is not None
    assert "Claim with citation. [1]" in answer_text
    assert "## Источники" in answer_text


def test_render_generated_answer_cohesive_text_with_filtered_sources() -> None:
    documents = [
        Document(
            page_content="Debezium SQL config",
            metadata={
                "source_type": "confluence",
                "document_type": "page",
                "document_id": "confluence:page:131304166",
                "chunk_id": "chunk-debezium",
                "page_id": "131304166",
                "title": "Debezium setup",
                "path": "Data/Debezium",
                "source": "https://wiki.example/pages/viewpage.action?pageId=131304166",
                "source_updated_at": "2026-06-06T10:00:00+00:00",
                "ingested_at": "2026-06-06T11:00:00+00:00",
            },
            id="chunk-debezium",
        ),
        Document(
            page_content="Kafka topic config",
            metadata={
                "source_type": "confluence",
                "document_type": "page",
                "document_id": "confluence:page:131304999",
                "chunk_id": "chunk-kafka",
                "page_id": "131304999",
                "title": "Kafka setup",
                "path": "Data/Kafka",
                "source": "https://wiki.example/pages/viewpage.action?pageId=131304999",
                "source_updated_at": "2026-06-06T10:00:00+00:00",
                "ingested_at": "2026-06-06T11:00:00+00:00",
            },
            id="chunk-kafka",
        ),
    ]
    context = build_citation_context(documents)

    answer_text, cited_answer = render_generated_answer(
        GeneratedAnswer(
            answer_text="Debezium connector is configured via SQL.",
            claim_specs=[("Debezium SQL config.", ["chunk-debezium"])],
        ),
        context,
    )

    assert cited_answer is not None
    assert len(cited_answer.sources) == 1
    assert "Debezium connector is configured via SQL." in answer_text
    assert "## Источники" in answer_text
    assert "Debezium setup" in answer_text
    assert "Kafka setup" not in answer_text
    assert "Debezium SQL config. [1]" not in answer_text


def test_smoke_cases_from_metrics_doc() -> None:
    smoke_cases = [
        {
            "name": "Простой вопрос без поиска",
            "query": "Привет!",
            "should_retrieve": False,
        },
        {
            "name": "Вопрос, требующий поиска",
            "query": "Что такое LangGraph?",
            "should_retrieve": True,
        },
        {
            "name": "Длинный вопрос",
            "query": "Объясни архитектуру LangGraph и её компоненты",
            "should_retrieve": True,
        },
    ]

    for case in smoke_cases:
        retriever_calls: list[str] = []

        def fake_retriever(query: str, *, _case=case) -> list[Document]:
            retriever_calls.append(query)
            return [_page_doc(chunk_id="chunk-smoke", text=query)]

        def should_retrieve(question: str, *, _case=case) -> bool:
            return _case["should_retrieve"]

        workflow = build_agentic_rag_workflow(
            retriever=fake_retriever,
            should_retrieve=should_retrieve,
            grade_documents=lambda query, docs: RelevanceDecision(is_relevant=True),
            generate_answer=lambda question, documents, citation_context: GeneratedAnswer(
                answer_text=f"Answer for: {question}"
            ),
            rewrite_query=lambda question, query, documents: query,
        )

        state = workflow.invoke(
            {
                "question": case["query"],
                "query": case["query"],
                "messages": [],
                "documents": [],
                "citation_context": None,
                "relevance_decision": None,
                "answer": "",
                "cited_answer": None,
                "retry_count": 0,
                "max_retries": 2,
                "should_retrieve": False,
            }
        )

        assert state["answer"]
        if case["should_retrieve"]:
            assert retriever_calls == [case["query"]]
        else:
            assert retriever_calls == []
