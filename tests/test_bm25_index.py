from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rague.retrieval.bm25_index import Bm25ChunkIndex, tokenize


def _doc(
    *,
    chunk_id: str,
    text: str,
    page_id: str = "page-1",
    source: str = "https://example.test/page",
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "page_id": page_id,
            "source": source,
            "title": "Test page",
        },
        id=chunk_id,
    )


def test_tokenize_lowercases_russian_and_latin_tokens() -> None:
    assert tokenize("Debezium коннектор v2_1") == [
        "debezium",
        "коннектор",
        "v2_1",
    ]


def test_from_documents_empty_corpus_returns_no_results() -> None:
    pytest.importorskip("rank_bm25")
    index = Bm25ChunkIndex.from_documents([])

    assert index.corpus_size == 0
    assert index.search("debezium") == []


def test_search_ranks_exact_term_higher() -> None:
    pytest.importorskip("rank_bm25")
    index = Bm25ChunkIndex.from_documents(
        [
            _doc(chunk_id="chunk-kafka", text="Kafka topic configuration"),
            _doc(chunk_id="chunk-debezium", text="Debezium connector setup"),
        ]
    )

    results = index.search("Debezium connector", limit=2)

    assert len(results) == 2
    assert results[0].document.metadata["chunk_id"] == "chunk-debezium"
    assert results[0].score > results[1].score


def test_search_respects_limit_and_preserves_metadata() -> None:
    pytest.importorskip("rank_bm25")
    index = Bm25ChunkIndex.from_documents(
        [
            _doc(chunk_id=f"chunk-{index}", text=f"Debezium topic {index}")
            for index in range(5)
        ]
    )

    results = index.search("Debezium", limit=2)

    assert len(results) == 2
    assert results[0].document.metadata["chunk_id"].startswith("chunk-")
    assert results[0].document.metadata["page_id"] == "page-1"
    assert results[0].document.metadata["source"] == "https://example.test/page"


def test_refresh_from_store_rebuilds_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("rank_bm25")
    index = Bm25ChunkIndex.from_documents(
        [_doc(chunk_id="chunk-old", text="old debezium text")]
    )
    old_built_at = index.built_at

    class FakeStore:
        def scroll_chunks(self) -> list[Document]:
            return [_doc(chunk_id="chunk-new", text="new debezium connector")]

    refreshed = index.refresh_from_store(FakeStore())  # type: ignore[arg-type]

    assert refreshed is index
    assert index.corpus_size == 1
    assert index.documents[0].metadata["chunk_id"] == "chunk-new"
    assert index.built_at is not None
    assert old_built_at is not None
    assert index.built_at >= old_built_at
