from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rague.citations import build_citation_context
from rague.retrieval.bm25_index import Bm25ChunkIndex
from rague.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rague.retrieval.hybrid_reranker import (
    CandidateRecord,
    HybridRerankerRetriever,
    HybridRetrieverConfig,
    build_retrieval_metadata_filter,
    create_retriever_from_config,
    merge_and_dedup_candidates,
)
from rague.vectorstores.qdrant_store import QdrantChunkStore, ScoredDocument


def _doc(
    *,
    chunk_id: str | None = None,
    document_id: str | None = None,
    page_id: str | None = None,
    text: str = "sample text",
) -> Document:
    metadata = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "page_id": page_id,
        "title": "Test page",
        "path": "Parent/Test page",
        "source": "https://example.test/page",
    }
    return Document(page_content=text, metadata=metadata, id=chunk_id)


def _full_page_doc(*, chunk_id: str, text: str = "sample text") -> Document:
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


def test_merge_and_dedup_prefers_chunk_id() -> None:
    shared = _doc(chunk_id="chunk-1", document_id="doc-1", page_id="page-1")
    vector_results = [ScoredDocument(document=shared, score=0.91)]
    bm25_results = [ScoredDocument(document=shared, score=4.2)]

    merged = merge_and_dedup_candidates(vector_results, bm25_results)

    assert len(merged) == 1
    assert merged[0].vector_score == 0.91
    assert merged[0].bm25_score == 4.2
    assert merged[0].retrieval_sources == ["vector", "bm25"]


def test_merge_and_dedup_uses_document_id_when_chunk_missing() -> None:
    vector_results = [
        ScoredDocument(
            document=_doc(document_id="doc-1", page_id="page-1", text="a"),
            score=0.5,
        )
    ]
    bm25_results = [
        ScoredDocument(
            document=_doc(document_id="doc-1", page_id="page-1", text="b"),
            score=2.0,
        )
    ]

    merged = merge_and_dedup_candidates(vector_results, bm25_results)

    assert len(merged) == 1
    assert merged[0].retrieval_sources == ["vector", "bm25"]


def test_merge_and_dedup_keeps_distinct_chunks() -> None:
    vector_results = [
        ScoredDocument(document=_doc(chunk_id="chunk-1"), score=0.8)
    ]
    bm25_results = [
        ScoredDocument(document=_doc(chunk_id="chunk-2"), score=3.1)
    ]

    merged = merge_and_dedup_candidates(vector_results, bm25_results)

    assert len(merged) == 2
    chunk_ids = {record.document.metadata["chunk_id"] for record in merged}
    assert chunk_ids == {"chunk-1", "chunk-2"}


def test_build_retrieval_metadata_filter_omits_none_values() -> None:
    assert build_retrieval_metadata_filter(
        source_type="confluence",
        document_type=None,
        space=None,
        page_id=None,
        is_current=True,
    ) == {"source_type": "confluence", "is_current": True}

    assert build_retrieval_metadata_filter(
        source_type=None,
        document_type=None,
        space=None,
        page_id=None,
        is_current=None,
    ) is None


def test_refresh_bm25_index_rebuilds_lazy_index(monkeypatch) -> None:
    pytest.importorskip("rank_bm25")
    retriever = HybridRerankerRetriever.model_construct(
        store=MagicMock(),
        embedder=MagicMock(),
    )
    built_indexes: list[Bm25ChunkIndex] = []

    def fake_from_store(store, *, filter_=None) -> Bm25ChunkIndex:
        del store, filter_
        index = Bm25ChunkIndex.from_documents(
            [_doc(chunk_id="chunk-refreshed", text="Debezium connector")]
        )
        built_indexes.append(index)
        return index

    monkeypatch.setattr(Bm25ChunkIndex, "from_store", fake_from_store)

    first = retriever.refresh_bm25_index()
    second = retriever.refresh_bm25_index()

    assert first is second
    assert len(built_indexes) == 2
    assert retriever._bm25_index.corpus_size == 1


def test_rerank_orders_candidates_by_cross_encoder_score() -> None:
    retriever = HybridRerankerRetriever.model_construct(
        store=MagicMock(),
        embedder=MagicMock(),
        top_k=2,
    )
    mock_reranker = MagicMock(spec=CrossEncoderReranker)
    mock_reranker.score_pairs.return_value = [0.2, 0.9, 0.5]
    retriever._reranker = mock_reranker

    candidates = [
        CandidateRecord(document=_doc(chunk_id="chunk-1"), retrieval_sources=["vector"]),
        CandidateRecord(document=_doc(chunk_id="chunk-2"), retrieval_sources=["bm25"]),
        CandidateRecord(document=_doc(chunk_id="chunk-3"), retrieval_sources=["vector"]),
    ]

    reranked = retriever._rerank_candidates("test query", candidates)

    assert [item.document.metadata["chunk_id"] for item in reranked] == [
        "chunk-2",
        "chunk-3",
        "chunk-1",
    ]
    assert reranked[0].document.metadata["rerank_score"] == 0.9
    mock_reranker.score_pairs.assert_called_once()


def test_hybrid_retriever_end_to_end_flow(monkeypatch) -> None:
    pytest.importorskip("rank_bm25")

    shared = _doc(chunk_id="chunk-shared", text="Debezium connector setup")
    vector_only = _doc(chunk_id="chunk-vector", text="Kafka topic config")
    bm25_only = _doc(chunk_id="chunk-bm25", text="Debezium setup guide")

    store = MagicMock()
    store.search_similar.return_value = [
        ScoredDocument(document=shared, score=0.91),
        ScoredDocument(document=vector_only, score=0.72),
    ]

    bm25_index = Bm25ChunkIndex.from_documents(
        [
            shared,
            bm25_only,
            _doc(chunk_id="chunk-noise", text="unrelated content"),
        ]
    )

    def fake_from_store(store_arg, *, filter_=None):
        del store_arg, filter_
        return bm25_index

    monkeypatch.setattr(Bm25ChunkIndex, "from_store", fake_from_store)

    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]

    mock_reranker = MagicMock(spec=CrossEncoderReranker)
    mock_reranker.score_pairs.return_value = [0.95, 0.4, 0.7]

    retriever = HybridRerankerRetriever.model_construct(
        store=store,
        embedder=embedder,
        top_k=2,
        bm25_candidate_limit=5,
        vector_candidate_limit=5,
    )
    retriever._reranker = mock_reranker

    results = retriever.invoke("Debezium connector")

    assert len(results) == 2
    assert results[0].metadata["chunk_id"] == "chunk-shared"
    assert results[0].metadata["rerank_score"] == 0.95
    assert results[0].metadata["vector_score"] == 0.91
    assert results[0].metadata["bm25_score"] is not None
    assert results[0].metadata["retrieval_sources"] == ["vector", "bm25"]
    assert results[1].metadata["chunk_id"] == "chunk-bm25"
    store.search_similar.assert_called_once()


def test_retrieval_results_build_citation_context_without_warnings(monkeypatch) -> None:
    pytest.importorskip("rank_bm25")

    shared = _full_page_doc(chunk_id="chunk-shared", text="Debezium connector setup")
    vector_only = _full_page_doc(chunk_id="chunk-vector", text="Kafka topic config")
    bm25_only = _full_page_doc(chunk_id="chunk-bm25", text="Debezium setup guide")

    store = MagicMock()
    store.search_similar.return_value = [
        ScoredDocument(document=shared, score=0.91),
        ScoredDocument(document=vector_only, score=0.72),
    ]

    bm25_index = Bm25ChunkIndex.from_documents(
        [
            shared,
            bm25_only,
            _full_page_doc(chunk_id="chunk-noise", text="unrelated content"),
        ]
    )

    def fake_from_store(store_arg, *, filter_=None):
        del store_arg, filter_
        return bm25_index

    monkeypatch.setattr(Bm25ChunkIndex, "from_store", fake_from_store)

    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]

    mock_reranker = MagicMock(spec=CrossEncoderReranker)
    mock_reranker.score_pairs.return_value = [0.95, 0.4, 0.7]

    retriever = HybridRerankerRetriever.model_construct(
        store=store,
        embedder=embedder,
        top_k=2,
        bm25_candidate_limit=5,
        vector_candidate_limit=5,
    )
    retriever._reranker = mock_reranker

    results = retriever.invoke("Debezium connector")
    context = build_citation_context(results)

    assert context.warnings == []
    assert context.sources[0].title == "Debezium setup"
    assert context.sources[0].path == "Data/Debezium"
    assert context.sources[0].source == (
        "https://wiki.example/pages/viewpage.action?pageId=131304166"
    )
    assert context.refs_by_chunk_id["chunk-shared"].label == "[1]"
    assert results[0].metadata["title"] == "Debezium setup"
    assert results[0].metadata["path"] == "Data/Debezium"
    assert results[0].metadata["source_updated_at"] == "2026-06-06T10:00:00+00:00"
    assert results[0].metadata["ingested_at"] == "2026-06-06T11:00:00+00:00"


def test_vector_and_bm25_use_metadata_filter(monkeypatch) -> None:
    pytest.importorskip("rank_bm25")

    sentinel_filter = object()
    monkeypatch.setattr(
        "rague.retrieval.hybrid_reranker.build_metadata_filter",
        lambda conditions: sentinel_filter if conditions else None,
    )

    store = MagicMock()
    store.search_similar.return_value = []
    captured_filters: list[object | None] = []

    def fake_from_store(store_arg, *, filter_=None):
        del store_arg
        captured_filters.append(filter_)
        return Bm25ChunkIndex.from_documents([])

    monkeypatch.setattr(Bm25ChunkIndex, "from_store", fake_from_store)

    metadata_filter = {"source_type": "confluence", "is_current": True}
    retriever = HybridRerankerRetriever.model_construct(
        store=store,
        embedder=MagicMock(embed_query=MagicMock(return_value=[0.1])),
        metadata_filter=metadata_filter,
        bm25_candidate_limit=5,
        vector_candidate_limit=5,
    )
    retriever._reranker = MagicMock(spec=CrossEncoderReranker)
    retriever._reranker.score_pairs.return_value = []

    retriever.invoke("query")

    kwargs = store.search_similar.call_args.kwargs
    assert kwargs["metadata_filter"] == metadata_filter
    assert captured_filters == [sentinel_filter]


def _make_store_with_client(client: MagicMock) -> QdrantChunkStore:
    store = QdrantChunkStore.__new__(QdrantChunkStore)
    store.url = "http://localhost:6333"
    store.collection_name = "test_collection"
    store.vector_size = 384
    store.distance = "Cosine"
    store.hnsw_config = None
    store._client = client
    return store


def test_create_retriever_from_config_runs_health_check(monkeypatch) -> None:
    monkeypatch.delenv("RAGUE_SKIP_QDRANT_HEALTH_CHECK", raising=False)

    client = MagicMock()
    client.collection_exists.return_value = True
    store = _make_store_with_client(client)
    health_check = MagicMock()
    store.check_retrieval_ready = health_check

    monkeypatch.setattr(
        "rague.retrieval.hybrid_reranker.QdrantChunkStore",
        lambda **kwargs: store,
    )

    config = HybridRetrieverConfig(
        qdrant_url="http://localhost:6333",
        collection_name="test_collection",
        embedding_provider="deterministic",
        embedding_vector_size=384,
    )
    mock_reranker = MagicMock(spec=CrossEncoderReranker)

    retriever = create_retriever_from_config(config, reranker=mock_reranker)

    health_check.assert_called_once_with(require_non_empty=True)
    assert retriever._reranker is mock_reranker


def test_create_retriever_from_config_skips_health_check_when_env_set(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RAGUE_SKIP_QDRANT_HEALTH_CHECK", "1")

    client = MagicMock()
    store = _make_store_with_client(client)
    health_check = MagicMock()
    store.check_retrieval_ready = health_check

    monkeypatch.setattr(
        "rague.retrieval.hybrid_reranker.QdrantChunkStore",
        lambda **kwargs: store,
    )

    create_retriever_from_config(
        HybridRetrieverConfig(
            embedding_provider="deterministic",
            embedding_vector_size=384,
        )
    )

    health_check.assert_not_called()


def test_create_retriever_from_config_passes_reranker_batch_size(monkeypatch) -> None:
    monkeypatch.setenv("RAGUE_SKIP_QDRANT_HEALTH_CHECK", "1")

    client = MagicMock()
    store = _make_store_with_client(client)
    store.check_retrieval_ready = MagicMock()

    monkeypatch.setattr(
        "rague.retrieval.hybrid_reranker.QdrantChunkStore",
        lambda **kwargs: store,
    )

    captured_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_cross_encoder_reranker(*args, **kwargs):
        captured_calls.append((args, kwargs))
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = str(args[0]) if args else str(kwargs.get("model_name", ""))
        reranker.batch_size = kwargs.get("batch_size")
        reranker._encoder = MagicMock()
        return reranker

    monkeypatch.setattr(
        "rague.retrieval.hybrid_reranker.CrossEncoderReranker",
        fake_cross_encoder_reranker,
    )

    retriever = create_retriever_from_config(
        HybridRetrieverConfig(
            embedding_provider="deterministic",
            embedding_vector_size=384,
            reranker_batch_size=4,
        )
    )
    retriever._get_reranker()

    assert captured_calls == [
        (("BAAI/bge-reranker-v2-m3",), {"batch_size": 4})
    ]
    assert retriever._reranker.batch_size == 4
