"""Hybrid BM25 + vector retrieval with cross-encoder reranking."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rague.embeddings.base import TextEmbedder
from rague.embeddings.factory import create_embedder
from rague.retrieval.bm25_index import Bm25ChunkIndex
from rague.retrieval.cross_encoder_reranker import (
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    resolve_reranker_model,
)
from rague.vectorstores.qdrant_store import (
    DEFAULT_COLLECTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_FULL_SCAN_THRESHOLD,
    HnswIndexConfig,
    QdrantChunkStore,
    QdrantHealthError,
    ScoredDocument,
    build_metadata_filter,
)

DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_LIMIT = 50


@dataclass(frozen=True)
class HybridRetrieverConfig:
    """Runtime configuration for hybrid retrieval."""

    qdrant_url: str = "http://localhost:6333"
    collection_name: str = DEFAULT_COLLECTION
    top_k: int = DEFAULT_TOP_K
    bm25_candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    vector_candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH
    hnsw_m: int | None = None
    hnsw_ef_construct: int | None = None
    hnsw_full_scan_threshold: int = DEFAULT_HNSW_FULL_SCAN_THRESHOLD
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_batch_size: int | None = DEFAULT_RERANKER_BATCH_SIZE
    metadata_filter: dict[str, object] | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_vector_size: int | None = None


@dataclass
class CandidateRecord:
    """Merged retrieval candidate before reranking."""

    document: Document
    vector_score: float | None = None
    bm25_score: float | None = None
    retrieval_sources: list[str] = field(default_factory=list)


class HybridRerankerRetriever(BaseRetriever):
    """Retrieve chunks with parallel BM25 and vector search, then rerank."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: QdrantChunkStore
    embedder: TextEmbedder
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_batch_size: int | None = DEFAULT_RERANKER_BATCH_SIZE
    top_k: int = DEFAULT_TOP_K
    bm25_candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    vector_candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH
    metadata_filter: dict[str, object] | None = None
    _bm25_index: Bm25ChunkIndex | None = None
    _reranker: CrossEncoderReranker | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        candidates = self._collect_candidates(query)
        if not candidates:
            return []

        reranked = self._rerank_candidates(query, candidates)
        return [_to_result_document(record) for record in reranked[: self.top_k]]

    def _collect_candidates(self, query: str) -> list[CandidateRecord]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(self._vector_candidates, query)
            bm25_future = executor.submit(self._bm25_candidates, query)
            vector_results = vector_future.result()
            bm25_results = bm25_future.result()
        return merge_and_dedup_candidates(vector_results, bm25_results)

    def _vector_candidates(self, query: str) -> list[ScoredDocument]:
        query_vector = self.embedder.embed_query(query)
        return self.store.search_similar(
            query_vector,
            limit=self.vector_candidate_limit,
            hnsw_ef=self.hnsw_ef_search,
            metadata_filter=self.metadata_filter,
        )

    def refresh_bm25_index(self) -> Bm25ChunkIndex:
        """Rebuild the in-memory BM25 index from the current Qdrant corpus."""
        qdrant_filter = self._metadata_qdrant_filter()
        if self._bm25_index is None:
            self._bm25_index = Bm25ChunkIndex.from_store(
                self.store,
                filter_=qdrant_filter,
            )
        else:
            self._bm25_index.refresh_from_store(
                self.store,
                filter_=qdrant_filter,
            )
        return self._bm25_index

    def _bm25_candidates(self, query: str) -> list[ScoredDocument]:
        if self._bm25_index is None:
            self._bm25_index = Bm25ChunkIndex.from_store(
                self.store,
                filter_=self._metadata_qdrant_filter(),
            )
        return self._bm25_index.search(query, limit=self.bm25_candidate_limit)

    def _metadata_qdrant_filter(self) -> Any | None:
        return build_metadata_filter(self.metadata_filter)

    def _rerank_candidates(
        self,
        query: str,
        candidates: list[CandidateRecord],
    ) -> list[CandidateRecord]:
        if not candidates:
            return []

        pairs = [(query, candidate.document.page_content) for candidate in candidates]
        scores = self._get_reranker().score_pairs(pairs)
        reranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results: list[CandidateRecord] = []
        for candidate, score in reranked:
            candidate.document.metadata["rerank_score"] = float(score)
            results.append(candidate)
        return results

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                self.reranker_model,
                batch_size=self.reranker_batch_size,
            )
        return self._reranker


def merge_and_dedup_candidates(
    vector_results: list[ScoredDocument],
    bm25_results: list[ScoredDocument],
) -> list[CandidateRecord]:
    """Merge BM25 and vector hits and deduplicate by stable identifiers."""
    merged: dict[str, CandidateRecord] = {}

    for source, results in (("vector", vector_results), ("bm25", bm25_results)):
        for result in results:
            key = _dedup_key(result.document)
            if key not in merged:
                merged[key] = CandidateRecord(
                    document=result.document,
                    retrieval_sources=[source],
                )
            else:
                record = merged[key]
                if source not in record.retrieval_sources:
                    record.retrieval_sources.append(source)

            record = merged[key]
            if source == "vector":
                record.vector_score = result.score
            else:
                record.bm25_score = result.score

    return list(merged.values())


def build_retrieval_metadata_filter(
    *,
    source_type: str | None = "confluence",
    document_type: str | None = None,
    space: str | None = None,
    page_id: str | None = None,
    is_current: bool | None = True,
) -> dict[str, object] | None:
    """Build a simple metadata filter dict for hybrid retrieval."""
    conditions: dict[str, object] = {}
    if source_type is not None:
        conditions["source_type"] = source_type
    if document_type is not None:
        conditions["document_type"] = document_type
    if space is not None:
        conditions["space"] = space
    if page_id is not None:
        conditions["page_id"] = page_id
    if is_current is not None:
        conditions["is_current"] = is_current
    return conditions or None


def _dedup_key(document: Document) -> str:
    metadata = document.metadata
    if metadata.get("chunk_id"):
        return f"chunk:{metadata['chunk_id']}"
    if metadata.get("document_id"):
        return f"document:{metadata['document_id']}"
    if metadata.get("page_id"):
        return f"page:{metadata['page_id']}"
    return f"content:{hash(document.page_content)}"


def _to_result_document(candidate: CandidateRecord) -> Document:
    metadata = dict(candidate.document.metadata)
    metadata["vector_score"] = candidate.vector_score
    metadata["bm25_score"] = candidate.bm25_score
    metadata["retrieval_sources"] = list(candidate.retrieval_sources)
    return Document(
        page_content=candidate.document.page_content,
        metadata=metadata,
        id=candidate.document.id,
    )


def _should_skip_qdrant_health_check() -> bool:
    return os.getenv("RAGUE_SKIP_QDRANT_HEALTH_CHECK", "").strip() == "1"


def create_retriever_from_config(
    config: HybridRetrieverConfig,
    *,
    reranker: CrossEncoderReranker | None = None,
) -> HybridRerankerRetriever:
    """Build a retriever from explicit configuration."""
    embedder = create_embedder(
        provider=config.embedding_provider,
        model_name=config.embedding_model,
        vector_size=config.embedding_vector_size,
    )
    store = QdrantChunkStore(
        url=config.qdrant_url,
        collection_name=config.collection_name,
        vector_size=embedder.vector_size,
        hnsw_config=HnswIndexConfig(
            m=config.hnsw_m,
            ef_construct=config.hnsw_ef_construct,
            full_scan_threshold=config.hnsw_full_scan_threshold,
        ),
    )
    if not _should_skip_qdrant_health_check():
        store.check_retrieval_ready(require_non_empty=True)

    retriever = HybridRerankerRetriever(
        store=store,
        embedder=embedder,
        reranker_model=resolve_reranker_model(config.reranker_model),
        reranker_batch_size=config.reranker_batch_size,
        top_k=config.top_k,
        bm25_candidate_limit=config.bm25_candidate_limit,
        vector_candidate_limit=config.vector_candidate_limit,
        hnsw_ef_search=config.hnsw_ef_search,
        metadata_filter=config.metadata_filter,
    )
    if reranker is not None:
        retriever._reranker = reranker
    return retriever


def _config_from_env() -> HybridRetrieverConfig:
    return HybridRetrieverConfig(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        collection_name=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
        top_k=int(os.getenv("RETRIEVAL_TOP_K", str(DEFAULT_TOP_K))),
        bm25_candidate_limit=int(
            os.getenv("RETRIEVAL_BM25_LIMIT", str(DEFAULT_CANDIDATE_LIMIT))
        ),
        vector_candidate_limit=int(
            os.getenv("RETRIEVAL_VECTOR_LIMIT", str(DEFAULT_CANDIDATE_LIMIT))
        ),
        hnsw_ef_search=int(
            os.getenv("QDRANT_HNSW_EF_SEARCH", str(DEFAULT_HNSW_EF_SEARCH))
        ),
        hnsw_m=_optional_env_int("QDRANT_HNSW_M"),
        hnsw_ef_construct=_optional_env_int("QDRANT_HNSW_EF_CONSTRUCT"),
        hnsw_full_scan_threshold=int(
            os.getenv(
                "QDRANT_HNSW_FULL_SCAN_THRESHOLD",
                str(DEFAULT_HNSW_FULL_SCAN_THRESHOLD),
            )
        ),
        reranker_model=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        reranker_batch_size=_optional_env_int("RERANKER_BATCH_SIZE"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER"),
        embedding_model=os.getenv("EMBEDDING_MODEL"),
        embedding_vector_size=_optional_env_int("EMBEDDING_VECTOR_SIZE"),
    )


def create_retriever_from_env() -> HybridRerankerRetriever:
    """Build a retriever using environment variables."""
    return create_retriever_from_config(_config_from_env())


def _optional_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hybrid BM25 + vector retrieval with cross-encoder reranking."
    )
    parser.add_argument("query", help="User question or search query.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("RETRIEVAL_TOP_K", str(DEFAULT_TOP_K))),
    )
    parser.add_argument(
        "--collection-name",
        default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=_optional_env_int("RERANKER_BATCH_SIZE"),
        help="Batch size passed to CrossEncoder.predict() (default: sentence-transformers internal).",
    )
    parser.add_argument(
        "--bm25-limit",
        type=int,
        default=int(os.getenv("RETRIEVAL_BM25_LIMIT", str(DEFAULT_CANDIDATE_LIMIT))),
    )
    parser.add_argument(
        "--vector-limit",
        type=int,
        default=int(os.getenv("RETRIEVAL_VECTOR_LIMIT", str(DEFAULT_CANDIDATE_LIMIT))),
    )
    parser.add_argument(
        "--source-type",
        default="confluence",
        help="Metadata filter for source_type (default: confluence).",
    )
    parser.add_argument(
        "--document-type",
        default=None,
        help="Optional metadata filter for document_type.",
    )
    parser.add_argument(
        "--space",
        default=None,
        help="Optional metadata filter for Confluence space key.",
    )
    parser.add_argument(
        "--page-id",
        default=None,
        help="Optional metadata filter for page_id.",
    )
    parser.add_argument(
        "--current-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, filter to is_current=true chunks.",
    )
    parser.add_argument(
        "--refresh-bm25",
        action="store_true",
        help="Rebuild the in-memory BM25 index before running retrieval.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    metadata_filter = build_retrieval_metadata_filter(
        source_type=args.source_type,
        document_type=args.document_type,
        space=args.space,
        page_id=args.page_id,
        is_current=args.current_only,
    )
    config = replace(
        _config_from_env(),
        qdrant_url=args.qdrant_url,
        collection_name=args.collection_name,
        top_k=args.top_k,
        reranker_model=args.reranker_model,
        reranker_batch_size=args.reranker_batch_size,
        bm25_candidate_limit=args.bm25_limit,
        vector_candidate_limit=args.vector_limit,
        metadata_filter=metadata_filter,
    )
    try:
        retriever = create_retriever_from_config(config)
    except QdrantHealthError as error:
        print(f"Retrieval setup failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    if args.refresh_bm25:
        retriever.refresh_bm25_index()
    documents = retriever.invoke(args.query)

    print(f"Query: {args.query}")
    print(f"Results: {len(documents)}\n")
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        print("=" * 72)
        print(f"RESULT {index}")
        print(f"title={metadata.get('title')}")
        print(f"chunk_id={metadata.get('chunk_id')}")
        print(f"rerank_score={metadata.get('rerank_score')}")
        print(f"vector_score={metadata.get('vector_score')}")
        print(f"bm25_score={metadata.get('bm25_score')}")
        print(f"sources={metadata.get('retrieval_sources')}")
        print("-" * 72)
        preview = document.page_content[:500]
        print(preview)
        if len(document.page_content) > 500:
            print("...")
        print()


if __name__ == "__main__":
    main()
