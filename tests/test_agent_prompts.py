from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rague.agents.prompts import (
    build_chat_prompt,
    format_documents_context,
    load_prompt_config,
    render_document_content_for_context,
)


def test_load_prompt_config_current_version() -> None:
    prompt = load_prompt_config("should_retrieve")

    assert "retrieval" in prompt.system.lower()
    assert "{question}" in prompt.user
    assert prompt.input_variables == ["question"]


def test_build_chat_prompt_generate_answer_formats_messages() -> None:
    prompt = build_chat_prompt("generate_answer")
    messages = prompt.format_messages(
        question="Что такое LangGraph?",
        documents_context="chunk context",
        allowed_chunk_ids="chunk-1",
    )

    assert len(messages) == 2
    assert "LangGraph" in messages[1].content
    assert "chunk-1" in messages[1].content


def test_missing_prompt_task_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="missing_task"):
        load_prompt_config("missing_task")


def test_format_documents_context_truncates_long_content() -> None:
    document = Document(
        page_content="x" * 2000,
        metadata={
            "chunk_id": "chunk-1",
            "title": "Title",
            "path": "Data/Path",
            "source": "https://example.test/page",
        },
        id="chunk-1",
    )

    rendered = format_documents_context([document], max_chars_per_doc=100)

    assert "chunk_id=chunk-1" in rendered
    assert "..." in rendered
    assert len(rendered) < 2000


def test_render_document_content_for_context_expands_code_summary() -> None:
    code_ref = "confluence:page:131304575:v8:code:1"
    sql = "select guid from ods.t_010_or_organization_common"
    document = Document(
        page_content=f"Тип: SQL\nПолный код: {code_ref}",
        metadata={
            "chunk_type": "code_summary",
            "code_language": "sql",
            "raw_code": sql,
            "chunk_id": code_ref,
        },
        id=code_ref,
    )

    rendered = render_document_content_for_context(document)

    assert "full_code:" in rendered
    assert "```sql" in rendered
    assert sql in rendered
    assert code_ref in rendered


def test_format_documents_context_expands_code_summary_with_metadata() -> None:
    code_ref = "confluence:page:131304575:v8:code:2"
    sql = "select table_name from meta.dq_log_error"
    document = Document(
        page_content=f"Тип: SQL\nПолный код: {code_ref}",
        metadata={
            "chunk_type": "code_summary",
            "code_language": "sql",
            "raw_code": sql,
            "chunk_id": code_ref,
            "title": "Asmodeus DQ",
            "path": "Data/DQ",
            "source": "https://wiki.example/page",
            "rerank_score": 0.91,
        },
        id=code_ref,
    )

    rendered = format_documents_context([document])

    assert f"chunk_id={code_ref}" in rendered
    assert "title=Asmodeus DQ" in rendered
    assert "rerank_score=0.9100" in rendered
    assert sql in rendered
    assert "meta.dq_log_error" in rendered


def test_format_documents_context_keeps_plain_text_unchanged() -> None:
    document = Document(
        page_content="plain text chunk",
        metadata={
            "chunk_type": "text",
            "chunk_id": "confluence:page:1:v1:chunk:0",
            "title": "Title",
            "path": "Data/Path",
            "source": "https://example.test/page",
        },
        id="confluence:page:1:v1:chunk:0",
    )

    rendered = format_documents_context([document])

    assert "plain text chunk" in rendered
    assert "full_code:" not in rendered


def test_format_documents_context_truncates_expanded_code_summary() -> None:
    code_ref = "confluence:page:131304575:v8:code:3"
    document = Document(
        page_content="Тип: SQL",
        metadata={
            "chunk_type": "code_summary",
            "code_language": "sql",
            "raw_code": "x" * 2000,
            "chunk_id": code_ref,
        },
        id=code_ref,
    )

    rendered = format_documents_context([document], max_chars_per_doc=100)

    assert "..." in rendered
    assert len(rendered) < 2000


def test_format_documents_context_without_rerank_score() -> None:
    document = Document(
        page_content="sample",
        metadata={
            "chunk_id": "chunk-2",
            "title": "Title",
            "path": "Data/Path",
            "source": "https://example.test/page2",
        },
        id="chunk-2",
    )

    rendered = format_documents_context([document])

    assert "chunk_id=chunk-2" in rendered
    assert "rerank_score=n/a" in rendered
