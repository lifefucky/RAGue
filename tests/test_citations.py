from __future__ import annotations

from langchain_core.documents import Document

from rague.citations import (
    build_citation_context,
    build_citation_sources,
    build_cited_answer_from_claim_specs,
    cite_claim,
    format_answer_with_sources,
    format_cited_answer_markdown,
    format_sources_markdown,
    missing_page_citation_fields,
)


def _page_doc(
    *,
    chunk_id: str,
    text: str = "sample",
    extra_metadata: dict | None = None,
) -> Document:
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
    if extra_metadata:
        metadata.update(extra_metadata)
    return Document(page_content=text, metadata=metadata, id=chunk_id)


def test_build_citation_sources_page_document() -> None:
    documents = [_page_doc(chunk_id="chunk-1")]

    sources, warnings = build_citation_sources(documents)

    assert len(sources) == 1
    assert len(warnings) == 0
    assert sources[0].source_id == "confluence:page:131304166"
    assert sources[0].document_type == "page"
    assert sources[0].chunk_ids == ["chunk-1"]
    assert sources[0].label == "[1]"
    assert sources[0].title == "Debezium setup"
    assert sources[0].path == "Data/Debezium"
    assert sources[0].source == "https://wiki.example/pages/viewpage.action?pageId=131304166"
    assert sources[0].metadata["citation_target"]["target_type"] == "page"


def test_build_citation_sources_dedup_same_page_chunks() -> None:
    documents = [
        _page_doc(chunk_id="chunk-1", text="first"),
        _page_doc(chunk_id="chunk-2", text="second"),
    ]

    sources, warnings = build_citation_sources(documents)

    assert len(sources) == 1
    assert len(warnings) == 0
    assert sources[0].chunk_ids == ["chunk-1", "chunk-2"]


def test_missing_page_citation_fields_detects_path_gap() -> None:
    metadata = {
        "source_type": "confluence",
        "document_type": "page",
        "document_id": "confluence:page:1",
        "chunk_id": "chunk-1",
        "page_id": "1",
        "title": "Title",
        "source": "https://wiki.example/page/1",
        "source_updated_at": "2026-06-06T10:00:00+00:00",
        "ingested_at": "2026-06-06T11:00:00+00:00",
    }

    missing = missing_page_citation_fields(metadata)

    assert missing == ["path"]


def test_build_citation_sources_page_missing_path_warns() -> None:
    documents = [
        Document(
            page_content="no path",
            metadata={
                "source_type": "confluence",
                "document_type": "page",
                "document_id": "confluence:page:1",
                "chunk_id": "chunk-1",
                "page_id": "1",
                "title": "Title",
                "source": "https://wiki.example/page/1",
                "source_updated_at": "2026-06-06T10:00:00+00:00",
                "ingested_at": "2026-06-06T11:00:00+00:00",
            },
            id="chunk-1",
        )
    ]

    _sources, warnings = build_citation_sources(documents)

    assert len(warnings) == 1
    assert "path" in warnings[0].missing_fields


def test_build_citation_context_creates_refs_by_chunk_id() -> None:
    documents = [
        _page_doc(chunk_id="chunk-1"),
        _page_doc(chunk_id="chunk-2"),
    ]

    context = build_citation_context(documents)

    assert len(context.sources) == 1
    assert context.sources[0].label == "[1]"
    assert set(context.refs_by_chunk_id) == {"chunk-1", "chunk-2"}
    assert context.refs_by_chunk_id["chunk-1"].label == "[1]"
    assert context.refs_by_chunk_id["chunk-1"].chunk_id == "chunk-1"


def test_build_citation_sources_preserves_code_metadata() -> None:
    documents = [
        _page_doc(
            chunk_id="chunk-code",
            extra_metadata={
                "chunk_type": "code_summary",
                "code_ref": "confluence:page:131304166:v1:code:0",
                "code_language": "sql",
                "caption": "DDL script",
            },
        )
    ]

    sources, _warnings = build_citation_sources(documents)

    assert sources[0].metadata["chunk_type"] == "code_summary"
    assert sources[0].metadata["code_ref"] == "confluence:page:131304166:v1:code:0"
    assert sources[0].metadata["code_language"] == "sql"
    assert sources[0].metadata["citation_target"]["target_type"] == "code"
    assert sources[0].metadata["citation_target"]["caption"] == "DDL script"


def test_build_citation_context_preserves_per_chunk_code_target_after_dedup() -> None:
    documents = [
        _page_doc(chunk_id="chunk-text", text="plain text"),
        _page_doc(
            chunk_id="chunk-code",
            text="code summary",
            extra_metadata={
                "chunk_type": "code_summary",
                "code_ref": "confluence:page:131304166:v1:code:0",
                "code_language": "sql",
                "caption": "DDL script",
            },
        ),
    ]

    context = build_citation_context(documents)

    assert len(context.sources) == 1
    code_ref = context.refs_by_chunk_id["chunk-code"]
    assert code_ref.metadata["citation_target"]["target_type"] == "code"
    assert code_ref.metadata["citation_target"]["code_ref"] == (
        "confluence:page:131304166:v1:code:0"
    )
    assert code_ref.metadata["citation_target"]["code_language"] == "sql"
    assert code_ref.metadata["citation_target"]["caption"] == "DDL script"


def test_build_citation_sources_preserves_attachment_like_metadata() -> None:
    documents = [
        Document(
            page_content="attachment chunk",
            metadata={
                "source_type": "confluence",
                "document_type": "attachment",
                "document_id": "confluence:page:1:attachment:a1",
                "chunk_id": "chunk-attachment-1",
                "title": "schema.pdf",
                "source": "https://wiki.example/download/attachments/1/schema.pdf",
                "path": "Data/Schema",
                "attachment_id": "a1",
                "attachment_filename": "schema.pdf",
                "attachment_media_type": "application/pdf",
                "attachment_version": 3,
                "attachment_updated_at": "2026-06-05T09:00:00+00:00",
                "source_updated_at": "2026-06-05T09:00:00+00:00",
                "parent_page_id": "1",
            },
            id="chunk-attachment-1",
        )
    ]

    context = build_citation_context(documents)
    sources = context.sources
    warnings = context.warnings

    assert len(sources) == 1
    assert len(warnings) == 0
    assert sources[0].document_type == "attachment"
    assert sources[0].metadata["attachment_id"] == "a1"
    assert sources[0].metadata["attachment_filename"] == "schema.pdf"
    assert sources[0].metadata["attachment_media_type"] == "application/pdf"
    assert sources[0].metadata["attachment_version"] == 3
    assert sources[0].metadata["attachment_updated_at"] == "2026-06-05T09:00:00+00:00"
    assert sources[0].metadata["source_updated_at"] == "2026-06-05T09:00:00+00:00"
    assert sources[0].metadata["parent_page_id"] == "1"
    assert sources[0].metadata["citation_target"]["target_type"] == "attachment"

    attachment_ref = context.refs_by_chunk_id["chunk-attachment-1"]
    assert attachment_ref.metadata["attachment_media_type"] == "application/pdf"
    assert attachment_ref.metadata["citation_target"]["attachment_id"] == "a1"


def test_build_citation_sources_missing_source_returns_warning() -> None:
    documents = [
        Document(
            page_content="no link",
            metadata={
                "document_type": "page",
                "document_id": "confluence:page:99",
                "chunk_id": "chunk-99",
                "title": "Untitled",
            },
            id="chunk-99",
        )
    ]

    sources, warnings = build_citation_sources(documents)

    assert len(sources) == 1
    assert len(warnings) == 1
    assert warnings[0].chunk_id == "chunk-99"
    assert "source" in warnings[0].missing_fields
    assert sources[0].source is None


def test_cite_claim_links_claim_to_chunk() -> None:
    context = build_citation_context([_page_doc(chunk_id="chunk-1")])

    claim, warnings = cite_claim(
        "Debezium connector is configured via SQL.",
        ["chunk-1"],
        context,
    )

    assert len(warnings) == 0
    assert len(claim.citation_refs) == 1
    assert claim.citation_refs[0].chunk_id == "chunk-1"
    assert claim.citation_refs[0].label == "[1]"


def test_cite_claim_unknown_chunk_id_warns() -> None:
    context = build_citation_context([_page_doc(chunk_id="chunk-1")])

    claim, warnings = cite_claim(
        "Unknown reference.",
        ["missing-chunk"],
        context,
    )

    assert claim.citation_refs == []
    assert len(warnings) == 1
    assert "missing-chunk" in warnings[0].message


def _distinct_page_doc(
    *,
    chunk_id: str,
    document_id: str,
    page_id: str,
    title: str,
    source: str,
    path: str,
    text: str = "sample",
) -> Document:
    metadata = {
        "source_type": "confluence",
        "document_type": "page",
        "document_id": document_id,
        "chunk_id": chunk_id,
        "page_id": page_id,
        "title": title,
        "path": path,
        "source": source,
        "source_updated_at": "2026-06-06T10:00:00+00:00",
        "ingested_at": "2026-06-06T11:00:00+00:00",
    }
    return Document(page_content=text, metadata=metadata, id=chunk_id)


def test_build_cited_answer_from_claim_specs_filters_uncited_sources() -> None:
    context = build_citation_context(
        [
            _distinct_page_doc(
                chunk_id="chunk-debezium",
                document_id="confluence:page:131304166",
                page_id="131304166",
                title="Debezium setup",
                source="https://wiki.example/pages/viewpage.action?pageId=131304166",
                path="Data/Debezium",
            ),
            _distinct_page_doc(
                chunk_id="chunk-kafka",
                document_id="confluence:page:131304999",
                page_id="131304999",
                title="Kafka setup",
                source="https://wiki.example/pages/viewpage.action?pageId=131304999",
                path="Data/Kafka",
            ),
        ]
    )

    answer = build_cited_answer_from_claim_specs(
        [("Debezium connector is configured via SQL.", ["chunk-debezium"])],
        context,
    )

    assert len(answer.sources) == 1
    assert answer.sources[0].title == "Debezium setup"
    rendered = format_cited_answer_markdown(answer)
    assert "Debezium setup" in rendered
    assert "Kafka setup" not in rendered


def test_format_cited_answer_markdown_renders_claim_markers_and_sources() -> None:
    context = build_citation_context(
        [
            _page_doc(chunk_id="chunk-1"),
            _page_doc(chunk_id="chunk-2"),
        ]
    )
    answer = build_cited_answer_from_claim_specs(
        [
            ("Debezium connector is configured via SQL.", ["chunk-1"]),
            ("Kafka topic must exist before startup.", ["chunk-2"]),
        ],
        context,
    )

    rendered = format_cited_answer_markdown(answer)

    assert "Debezium connector is configured via SQL. [1]" in rendered
    assert "Kafka topic must exist before startup. [1]" in rendered
    assert "## Источники" in rendered
    assert "[Debezium setup](https://wiki.example/pages/viewpage.action?pageId=131304166)" in rendered
    assert "updated: 2026-06-06T10:00:00+00:00" in rendered


def test_format_sources_markdown_renders_links() -> None:
    documents = [_page_doc(chunk_id="chunk-1")]
    sources, _warnings = build_citation_sources(documents)

    rendered = format_sources_markdown(sources)

    assert "## Источники" in rendered
    assert "[1] [Debezium setup](https://wiki.example/pages/viewpage.action?pageId=131304166)" in rendered
    assert "Data/Debezium" in rendered
    assert "chunks: chunk-1" in rendered


def test_format_sources_markdown_without_link_uses_plain_title() -> None:
    documents = [
        Document(
            page_content="no link",
            metadata={
                "document_type": "page",
                "document_id": "confluence:page:99",
                "chunk_id": "chunk-99",
                "title": "Untitled",
                "path": "Parent/Untitled",
            },
            id="chunk-99",
        )
    ]
    sources, _warnings = build_citation_sources(documents)

    rendered = format_sources_markdown(sources, include_chunk_ids=False)

    assert "Untitled — Parent/Untitled" in rendered
    assert "](" not in rendered


def test_format_answer_with_sources_combines_sections() -> None:
    documents = [_page_doc(chunk_id="chunk-1")]
    sources, _warnings = build_citation_sources(documents)

    rendered = format_answer_with_sources("Debezium connector is configured via SQL.", sources)

    assert rendered.startswith("Debezium connector is configured via SQL.")
    assert "## Источники" in rendered
