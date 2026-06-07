from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rague.vectorstores.qdrant_store import (
    PAYLOAD_INDEX_FIELDS,
    QdrantChunkStore,
    QdrantHealthError,
    _normalize_search_hits,
    _payload_to_document,
    build_metadata_filter,
)

pytestmark_integration = pytest.mark.skipif(
    os.getenv("RAGUE_RUN_QDRANT_INTEGRATION") != "1",
    reason="Set RAGUE_RUN_QDRANT_INTEGRATION=1 to run Qdrant integration tests.",
)


@dataclass
class FakeHit:
    payload: dict[str, Any]
    score: float


class FakeQueryPointsResponse:
    def __init__(self, points: list[FakeHit]) -> None:
        self.points = points


def _make_store(client: Any) -> QdrantChunkStore:
    store = QdrantChunkStore.__new__(QdrantChunkStore)
    store.url = "http://localhost:6333"
    store.collection_name = "test_collection"
    store.vector_size = 4
    store.distance = "Cosine"
    store.hnsw_config = None
    store._client = client
    return store


def test_payload_index_fields_include_citation_contract() -> None:
    required_fields = {
        "source_type",
        "document_type",
        "document_id",
        "chunk_id",
        "page_id",
        "title",
        "path",
        "source",
        "source_updated_at",
        "ingested_at",
    }

    assert required_fields.issubset(set(PAYLOAD_INDEX_FIELDS))


def test_build_metadata_filter_creates_must_conditions() -> None:
    pytest.importorskip("qdrant_client")
    query_filter = build_metadata_filter(
        {"source_type": "confluence", "is_current": True, "page_id": "131304166"}
    )

    assert query_filter is not None
    assert len(query_filter.must) == 3
    assert query_filter.must[0].key == "source_type"
    assert query_filter.must[0].match.value == "confluence"
    assert query_filter.must[1].match.value is True


def test_build_metadata_filter_returns_none_for_empty_input() -> None:
    assert build_metadata_filter(None) is None
    assert build_metadata_filter({}) is None
    pytest.importorskip("qdrant_client")
    assert build_metadata_filter({"page_id": None}) is None


def test_payload_to_document_maps_citation_fields() -> None:
    document = _payload_to_document(
        {
            "text": "Debezium connector setup",
            "chunk_id": "confluence:page:1:v1:chunk:0",
            "document_id": "confluence:page:1",
            "page_id": "1",
            "title": "Debezium",
            "path": "Parent/Debezium",
            "source": "https://example.test/page/1",
            "source_updated_at": "2026-06-06T10:00:00+00:00",
            "ingested_at": "2026-06-06T11:00:00+00:00",
        }
    )

    assert document.page_content == "Debezium connector setup"
    assert document.id == "confluence:page:1:v1:chunk:0"
    assert "text" not in document.metadata
    assert document.metadata["title"] == "Debezium"
    assert document.metadata["path"] == "Parent/Debezium"
    assert document.metadata["source"] == "https://example.test/page/1"


def test_payload_to_document_handles_missing_text() -> None:
    document = _payload_to_document({"chunk_id": "chunk-1", "title": "Empty"})

    assert document.page_content == ""
    assert document.id == "chunk-1"
    assert document.metadata == {"chunk_id": "chunk-1", "title": "Empty"}


def test_normalize_search_hits_handles_empty_payload() -> None:
    hits = _normalize_search_hits([FakeHit(payload={}, score=0.42)])

    assert hits == [({}, 0.42)]


def test_check_retrieval_ready_raises_when_qdrant_unreachable() -> None:
    client = MagicMock()
    client.get_collections.side_effect = ConnectionError("connection refused")
    store = _make_store(client)

    with pytest.raises(QdrantHealthError, match="not reachable"):
        store.check_retrieval_ready()


def test_check_retrieval_ready_raises_when_collection_missing() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock()
    client.collection_exists.return_value = False
    store = _make_store(client)

    with pytest.raises(QdrantHealthError, match="was not found"):
        store.check_retrieval_ready()


def test_check_retrieval_ready_raises_when_collection_empty() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock()
    client.collection_exists.return_value = True
    client.count.return_value = MagicMock(count=0)
    store = _make_store(client)

    with pytest.raises(QdrantHealthError, match="is empty"):
        store.check_retrieval_ready(require_non_empty=True)


def test_check_retrieval_ready_succeeds_for_existing_collection() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock()
    client.collection_exists.return_value = True
    store = _make_store(client)

    store.check_retrieval_ready()


def test_search_similar_uses_query_points_when_available() -> None:
    pytest.importorskip("qdrant_client")
    client = MagicMock()
    client.query_points.return_value = FakeQueryPointsResponse(
        [
            FakeHit(
                payload={
                    "text": "vector hit",
                    "chunk_id": "chunk-vector",
                    "title": "Vector page",
                },
                score=0.91,
            )
        ]
    )

    store = _make_store(client)
    results = store.search_similar(
        [1.0, 0.0, 0.0, 0.0],
        limit=5,
        score_threshold=0.5,
        metadata_filter={"source_type": "confluence"},
        hnsw_ef=64,
    )

    assert len(results) == 1
    assert results[0].score == 0.91
    assert results[0].document.page_content == "vector hit"
    assert results[0].document.id == "chunk-vector"
    client.query_points.assert_called_once()
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["query"] == [1.0, 0.0, 0.0, 0.0]
    assert kwargs["limit"] == 5
    assert kwargs["score_threshold"] == 0.5
    assert kwargs["with_payload"] is True
    assert kwargs["with_vectors"] is False
    assert kwargs["search_params"].hnsw_ef == 64
    assert kwargs["query_filter"].must[0].key == "source_type"
    client.search.assert_not_called()


def test_search_similar_falls_back_to_search() -> None:
    pytest.importorskip("qdrant_client")
    client = MagicMock()
    client.query_points.side_effect = TypeError("unsupported signature")
    client.search.return_value = [
        FakeHit(
            payload={
                "text": "legacy search hit",
                "chunk_id": "chunk-legacy",
            },
            score=0.77,
        )
    ]

    store = _make_store(client)
    results = store.search_similar([0.0, 1.0, 0.0, 0.0], limit=3)

    assert len(results) == 1
    assert results[0].score == 0.77
    assert results[0].document.page_content == "legacy search hit"
    client.search.assert_called_once()
    kwargs = client.search.call_args.kwargs
    assert kwargs["query_vector"] == [0.0, 1.0, 0.0, 0.0]
    assert kwargs["limit"] == 3


@pytestmark_integration
def test_search_similar_integration_with_live_qdrant() -> None:
    pytest.importorskip("qdrant_client")

    from qdrant_client import QdrantClient

    collection_name = f"test_vector_search_{uuid.uuid4().hex[:8]}"
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    store = QdrantChunkStore(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        collection_name=collection_name,
        vector_size=4,
    )
    store._client = client

    try:
        store.ensure_collection()
        chunks = [
            Document(
                page_content="Debezium connector setup",
                metadata={
                    "chunk_id": "chunk-debezium",
                    "document_id": "doc-debezium",
                    "page_id": "page-1",
                    "title": "Debezium",
                    "path": "Data/Debezium",
                    "source": "https://example.test/debezium",
                    "source_type": "confluence",
                    "is_current": True,
                },
                id="chunk-debezium",
            ),
            Document(
                page_content="Kafka topic configuration",
                metadata={
                    "chunk_id": "chunk-kafka",
                    "document_id": "doc-kafka",
                    "page_id": "page-2",
                    "title": "Kafka",
                    "path": "Data/Kafka",
                    "source": "https://example.test/kafka",
                    "source_type": "confluence",
                    "is_current": True,
                },
                id="chunk-kafka",
            ),
        ]
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        store.upsert_chunks(chunks, vectors)

        results = store.search_similar(
            [0.95, 0.05, 0.0, 0.0],
            limit=1,
            metadata_filter={"source_type": "confluence", "is_current": True},
            hnsw_ef=32,
        )

        assert len(results) == 1
        assert results[0].document.id == "chunk-debezium"
        assert results[0].document.page_content == "Debezium connector setup"
        assert results[0].document.metadata["title"] == "Debezium"
        assert results[0].score > 0.5
    finally:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
