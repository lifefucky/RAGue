from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rague.agents.prompts import (
    build_chat_prompt,
    format_documents_context,
    load_prompt_config,
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
