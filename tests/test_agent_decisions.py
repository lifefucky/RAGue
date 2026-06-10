from __future__ import annotations

from langchain_core.documents import Document

from rague.agents.decisions import AgentLlmDecisions
from rague.agents.parsers import (
    DocumentRelevanceOutput,
    GeneratedAnswerOutput,
    RewriteQueryOutput,
    ShouldRetrieveOutput,
    ClaimOutput,
)
from rague.citations import build_citation_context


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response

    def invoke(self, messages):
        del messages
        return self.response


class FakeChatModel:
    def __init__(self, responses):
        self.responses = list(responses)

    def with_structured_output(self, output_model):
        del output_model
        return FakeStructuredModel(self.responses.pop(0))

    def invoke(self, messages):
        del messages
        raise AssertionError("Plain invoke should not be used when structured output works.")


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


def test_decide_should_retrieve_uses_structured_output() -> None:
    decisions = AgentLlmDecisions(
        FakeChatModel([ShouldRetrieveOutput(needs_retrieval=False, reason="greeting")])
    )

    assert decisions.decide_should_retrieve("Привет!") is False


def test_grade_documents_returns_relevance_decision() -> None:
    decisions = AgentLlmDecisions(
        FakeChatModel([DocumentRelevanceOutput(is_relevant=True, reason="ok")])
    )

    decision = decisions.grade_documents("query", [_page_doc(chunk_id="chunk-1")])

    assert decision.is_relevant is True
    assert decision.reason == "ok"


def test_rewrite_query_returns_new_query() -> None:
    decisions = AgentLlmDecisions(
        FakeChatModel([RewriteQueryOutput(query="refined query", reason="better")])
    )

    rewritten = decisions.rewrite_query(
        "question",
        "query",
        [_page_doc(chunk_id="chunk-1")],
    )

    assert rewritten == "refined query"


def test_generate_answer_receives_expanded_code_context() -> None:
    code_ref = "confluence:page:131304575:v8:code:1"
    sql = "select guid from meta.dq_log_error"
    document = Document(
        page_content=f"Тип: SQL\nПолный код: {code_ref}",
        metadata={
            "source_type": "confluence",
            "document_type": "page",
            "document_id": "confluence:page:131304575",
            "chunk_type": "code_summary",
            "code_language": "sql",
            "raw_code": sql,
            "chunk_id": code_ref,
            "page_id": "131304575",
            "title": "Asmodeus DQ",
            "path": "Data/DQ",
            "source": "https://wiki.example/pages/viewpage.action?pageId=131304575",
            "source_updated_at": "2026-06-06T10:00:00+00:00",
            "ingested_at": "2026-06-06T11:00:00+00:00",
        },
        id=code_ref,
    )
    context = build_citation_context([document])
    captured_messages: list[object] = []

    class CapturingChatModel:
        def with_structured_output(self, output_model):
            del output_model
            return self

        def invoke(self, messages):
            captured_messages.append(messages)
            return GeneratedAnswerOutput(
                claims=[ClaimOutput(text="SQL script", chunk_ids=[code_ref])]
            )

    decisions = AgentLlmDecisions(CapturingChatModel())
    generated = decisions.generate_answer("Какой script для meta.dq_log_error?", [document], context)

    assert generated.claim_specs == [("SQL script", [code_ref])]
    assert captured_messages
    human_content = captured_messages[0][-1].content
    assert "full_code:" in human_content
    assert sql in human_content


def test_generate_answer_returns_claim_specs_and_filters_unknown_ids() -> None:
    documents = [_page_doc(chunk_id="chunk-1")]
    context = build_citation_context(documents)
    decisions = AgentLlmDecisions(
        FakeChatModel(
            [
                GeneratedAnswerOutput(
                    claims=[
                        ClaimOutput(text="Claim.", chunk_ids=["chunk-1", "missing"]),
                    ]
                )
            ]
        )
    )

    generated = decisions.generate_answer("question", documents, context)

    assert generated.claim_specs == [("Claim.", ["chunk-1"])]
