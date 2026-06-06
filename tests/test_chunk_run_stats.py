from langchain_core.documents import Document

from rague.ingestion.changelog import ChunkRunStats, IngestionRunReport


def _chunk(chunk_type: str, text: str = "sample") -> Document:
    return Document(page_content=text, metadata={"chunk_type": chunk_type})


def test_record_page_aggregates_by_type_and_counts() -> None:
    stats = ChunkRunStats()
    stats.record_page(
        [
            _chunk("text", "a" * 10),
            _chunk("code", "b" * 20),
            _chunk("code_summary"),
            _chunk("table_row"),
        ]
    )

    assert stats.total == 4
    assert stats.chars_total == 42
    assert stats.by_type == {
        "text": 1,
        "code": 1,
        "code_summary": 1,
        "table_row": 1,
    }
    assert stats.avg_per_page == 4.0
    assert stats.min_per_page == 4
    assert stats.max_per_page == 4
    assert stats.code_fragments_total == 2
    assert stats.format_code_fragments() == "total=2 (code=1, code_summary=1)"


def test_format_code_fragments_zero_when_no_code() -> None:
    stats = ChunkRunStats()
    stats.record_page([_chunk("text"), _chunk("table_row")])

    assert stats.code_fragments_total == 0
    assert stats.format_code_fragments() == "total=0"


def test_record_page_skips_empty_chunk_list() -> None:
    stats = ChunkRunStats()
    stats.record_page([])

    assert stats.total == 0
    assert stats.per_page_counts == []
    assert stats.min_per_page is None
    assert stats.max_per_page is None


def test_record_page_tracks_min_max_across_pages() -> None:
    stats = ChunkRunStats()
    stats.record_page([_chunk("text")])
    stats.record_page([_chunk("text"), _chunk("code"), _chunk("code")])

    assert stats.total == 4
    assert stats.avg_per_page == 2.0
    assert stats.min_per_page == 1
    assert stats.max_per_page == 3
    assert stats.format_by_type() == "code=2, text=2"


def test_ingestion_report_renders_chunk_summary() -> None:
    report = IngestionRunReport(
        scope="page_ids=['1']",
        collection_name="test",
        embedding_provider="deterministic",
        embedding_model="deterministic-hash",
        duration_seconds=12.5,
    )
    report.chunk_stats.record_page([_chunk("text"), _chunk("code")])
    markdown = report._render_markdown()

    assert "## Chunk Summary" in markdown
    assert "Total chunks: `2`" in markdown
    assert "avg `2.0`" in markdown
    assert "`code`: `1`" in markdown
    assert "`text`: `1`" in markdown
    assert "Duration: `12.5s`" in markdown
    assert "Code fragments: `1` (`code`: `1`, `code_summary`: `0`)" in markdown
    assert "## Attachments" in markdown


def test_format_attachment_summary() -> None:
    report = IngestionRunReport(
        scope="page_ids=['1']",
        collection_name="test",
        embedding_provider="deterministic",
        embedding_model="deterministic-hash",
    )
    report.attachments_discovered = 18
    report.attachment_samples_saved = 4
    report.attachments_skipped = 14
    report.attachment_sample_extensions = ["pdf", "pptx", "pdf"]

    assert report.format_attachment_summary() == (
        "discovered=18, samples_saved=4, skipped=14, failed=0, extensions=pdf, pptx"
    )


def test_format_attachment_extensions_deduplicates_and_sorts() -> None:
    report = IngestionRunReport(
        scope="page_ids=['1']",
        collection_name="test",
        embedding_provider="deterministic",
        embedding_model="deterministic-hash",
    )
    report.attachment_sample_extensions = ["pptx", "pdf", "pdf"]

    assert report.format_attachment_extensions() == "pdf, pptx"


def test_print_summary_includes_code_fragments_and_attachments(capsys) -> None:
    report = IngestionRunReport(
        scope="page_ids=['1']",
        collection_name="test",
        embedding_provider="deterministic",
        embedding_model="deterministic-hash",
        duration_seconds=5.0,
    )
    report.chunk_stats.record_page([_chunk("code"), _chunk("code_summary")])
    report.attachments_discovered = 3
    report.attachment_samples_saved = 1
    report.attachment_sample_extensions = ["pdf"]

    report.print_summary()
    output = capsys.readouterr().out

    assert "Code fragments: total=2 (code=1, code_summary=1)" in output
    assert (
        "Attachments: discovered=3, samples_saved=1, skipped=0, failed=0, "
        "extensions=pdf"
    ) in output


def test_print_summary_zero_code_fragments(capsys) -> None:
    report = IngestionRunReport(
        scope="page_ids=['1']",
        collection_name="test",
        embedding_provider="deterministic",
        embedding_model="deterministic-hash",
        duration_seconds=1.0,
    )
    report.chunk_stats.record_page([_chunk("text")])

    report.print_summary()
    output = capsys.readouterr().out

    assert "Code fragments: total=0" in output
