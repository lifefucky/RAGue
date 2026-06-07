"""Qdrant vector store helpers for Confluence chunk ingestion and retrieval."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.documents import Document

DEFAULT_COLLECTION = "confluence_pages_v1"
DEFAULT_DISTANCE = "Cosine"
# Temporary default for small collections so Qdrant uses HNSW during retrieval
# tests. Revisit after recall/latency benchmarks on larger corpora.
DEFAULT_HNSW_FULL_SCAN_THRESHOLD = 10
DEFAULT_HNSW_EF_SEARCH = 128


class QdrantHealthError(RuntimeError):
    """Raised when Qdrant is unavailable or not ready for retrieval."""


@dataclass(frozen=True)
class HnswIndexConfig:
    """Optional Qdrant HNSW vector index settings."""

    m: int | None = None
    ef_construct: int | None = None
    full_scan_threshold: int | None = None


PAYLOAD_INDEX_FIELDS = (
    "source_type",
    "document_type",
    "document_id",
    "chunk_id",
    "page_id",
    "title",
    "path",
    "source",
    "space",
    "source_updated_at",
    "ingested_at",
    "parent_page_id",
    "is_current",
    "chunk_type",
    "code_language",
    "code_ref",
)


@dataclass
class UpsertStats:
    points_upserted: int = 0
    points_deleted: int = 0


@dataclass(frozen=True)
class ScoredDocument:
    """Chunk document with an associated retrieval score."""

    document: Document
    score: float


class QdrantChunkStore:
    """Manage Confluence chunk points in a Qdrant collection."""

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        collection_name: str = DEFAULT_COLLECTION,
        vector_size: int,
        distance: str = DEFAULT_DISTANCE,
        hnsw_config: HnswIndexConfig | None = None,
    ) -> None:
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.hnsw_config = hnsw_config
        self._client = _create_qdrant_client(url)

    def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        hnsw_config = _build_hnsw_config(self.hnsw_config)
        if not self._client.collection_exists(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=distance_map.get(self.distance, Distance.COSINE),
                    hnsw_config=hnsw_config,
                ),
            )
        elif hnsw_config is not None:
            self._client.update_collection(
                collection_name=self.collection_name,
                hnsw_config=hnsw_config,
            )

        self._ensure_payload_indexes()

    def check_retrieval_ready(self, *, require_non_empty: bool = False) -> None:
        """Verify Qdrant is reachable and the target collection exists."""
        try:
            self._client.get_collections()
        except Exception as error:
            message = (
                f"Qdrant is not reachable at {self.url}. "
                "Start local Qdrant with "
                "`docker compose -f docker-compose.qdrant.yml up -d` "
                "or verify QDRANT_URL."
            )
            raise QdrantHealthError(message) from error

        if not self._client.collection_exists(self.collection_name):
            message = (
                f"Qdrant collection `{self.collection_name}` was not found. "
                "Run ingestion first or verify QDRANT_COLLECTION."
            )
            raise QdrantHealthError(message)

        if require_non_empty:
            count_result = self._client.count(collection_name=self.collection_name)
            point_count = int(getattr(count_result, "count", 0))
            if point_count == 0:
                message = (
                    f"Qdrant collection `{self.collection_name}` is empty. "
                    "Run ingestion before retrieval."
                )
                raise QdrantHealthError(message)

    def get_max_source_updated_at(self) -> datetime | None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        records, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value="confluence"),
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        latest: datetime | None = None
        for record in records:
            payload = record.payload or {}
            parsed = _parse_datetime(payload.get("source_updated_at"))
            if parsed and (latest is None or parsed > latest):
                latest = parsed
        return latest

    def delete_by_page_id(self, page_id: str) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        selector = Filter(
            must=[
                FieldCondition(
                    key="page_id",
                    match=MatchValue(value=str(page_id)),
                )
            ]
        )
        result = self._client.delete(
            collection_name=self.collection_name,
            points_selector=selector,
        )
        return _extract_deleted_count(result)

    def upsert_chunks(
        self,
        chunks: list[Document],
        vectors: list[list[float]],
    ) -> int:
        if len(chunks) != len(vectors):
            message = "Number of chunks must match number of embedding vectors."
            raise ValueError(message)

        from qdrant_client.models import PointStruct

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            payload = _build_payload(chunk)
            point_id = _stable_point_uuid(payload["chunk_id"])
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        if not points:
            return 0

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(points)

    def scroll_chunks(
        self,
        *,
        limit: int = 256,
        filter_: Any | None = None,
    ) -> list[Document]:
        """Load chunk documents from Qdrant for lexical retrieval indexes."""
        documents: list[Document] = []
        offset = None

        while True:
            records, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break

            for record in records:
                payload = record.payload or {}
                documents.append(_payload_to_document(payload))

            if next_offset is None:
                break
            offset = next_offset

        return documents

    def search_similar(
        self,
        query_vector: list[float],
        *,
        limit: int = 50,
        score_threshold: float | None = None,
        filter_: Any | None = None,
        metadata_filter: dict[str, Any] | None = None,
        hnsw_ef: int | None = None,
    ) -> list[ScoredDocument]:
        """Run vector similarity search and return scored chunk documents."""
        resolved_filter = _resolve_search_filter(
            filter_=filter_,
            metadata_filter=metadata_filter,
        )
        hits = self._execute_vector_search(
            query_vector,
            limit=limit,
            score_threshold=score_threshold,
            filter_=resolved_filter,
            hnsw_ef=hnsw_ef,
        )
        return [
            ScoredDocument(
                document=_payload_to_document(payload),
                score=score,
            )
            for payload, score in hits
        ]

    def _execute_vector_search(
        self,
        query_vector: list[float],
        *,
        limit: int,
        score_threshold: float | None,
        filter_: Any | None,
        hnsw_ef: int | None,
    ) -> list[tuple[dict[str, Any], float]]:
        from qdrant_client.models import SearchParams

        search_params = SearchParams(hnsw_ef=hnsw_ef) if hnsw_ef is not None else None
        common_kwargs = {
            "collection_name": self.collection_name,
            "limit": limit,
            "score_threshold": score_threshold,
            "search_params": search_params,
            "with_payload": True,
            "with_vectors": False,
        }

        if hasattr(self._client, "query_points"):
            try:
                response = self._client.query_points(
                    query=query_vector,
                    query_filter=filter_,
                    **common_kwargs,
                )
                points = getattr(response, "points", response)
                return _normalize_search_hits(points)
            except TypeError:
                pass

        hits = self._client.search(
            query_vector=query_vector,
            query_filter=filter_,
            **common_kwargs,
        )
        return _normalize_search_hits(hits)

    def _ensure_payload_indexes(self) -> None:
        from qdrant_client.models import PayloadSchemaType

        for field_name in PAYLOAD_INDEX_FIELDS:
            try:
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # Index may already exist or field type may differ; ignore for MVP.
                continue


def enrich_chunk_metadata(chunk: Document) -> Document:
    """Ensure stable chunk identifiers required by Qdrant ingestion."""
    metadata = dict(chunk.metadata)
    page_id = str(metadata.get("page_id") or metadata.get("id", "unknown"))
    page_version = metadata.get("page_version") or metadata.get("version", "unknown")
    chunk_index = metadata.get("chunk_index", 0)

    document_id = metadata.get("document_id") or f"confluence:page:{page_id}"
    chunk_id = (
        metadata.get("chunk_id")
        or f"{document_id}:v{page_version}:chunk:{chunk_index}"
    )

    metadata["document_id"] = document_id
    metadata["chunk_id"] = chunk_id
    metadata["page_id"] = page_id
    metadata["page_version"] = page_version
    metadata["text"] = chunk.page_content
    metadata.setdefault("source_type", "confluence")
    metadata.setdefault("document_type", "page")
    metadata.setdefault("is_current", True)

    chunk.metadata = metadata
    chunk.id = chunk_id
    return chunk


def _build_payload(chunk: Document) -> dict[str, Any]:
    payload = dict(chunk.metadata)
    payload["text"] = chunk.page_content
    return payload


def build_metadata_filter(conditions: dict[str, Any] | None) -> Any | None:
    """Build a Qdrant payload filter from simple metadata key/value pairs."""
    if not conditions:
        return None

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    must: list[FieldCondition] = []
    for key, value in conditions.items():
        if value is None:
            continue
        must.append(
            FieldCondition(
                key=key,
                match=MatchValue(value=_normalize_filter_value(value)),
            )
        )
    if not must:
        return None
    return Filter(must=must)


def _resolve_search_filter(
    *,
    filter_: Any | None,
    metadata_filter: dict[str, Any] | None,
) -> Any | None:
    if filter_ is not None:
        return filter_
    return build_metadata_filter(metadata_filter)


def _normalize_filter_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    return str(value)


def _normalize_search_hits(hits: Any) -> list[tuple[dict[str, Any], float]]:
    normalized: list[tuple[dict[str, Any], float]] = []
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        score = float(getattr(hit, "score", 0.0))
        normalized.append((payload, score))
    return normalized


def _payload_to_document(payload: dict[str, Any]) -> Document:
    metadata = dict(payload)
    page_content = str(metadata.pop("text", "") or "")
    chunk_id = metadata.get("chunk_id")
    return Document(
        page_content=page_content,
        metadata=metadata,
        id=str(chunk_id) if chunk_id else None,
    )


def _stable_point_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _build_hnsw_config(config: HnswIndexConfig | None):
    if config is None:
        return None

    values = {
        "m": config.m,
        "ef_construct": config.ef_construct,
        "full_scan_threshold": config.full_scan_threshold,
    }
    configured_values = {
        key: value for key, value in values.items() if value is not None
    }
    if not configured_values:
        return None

    from qdrant_client.models import HnswConfigDiff

    return HnswConfigDiff(**configured_values)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _extract_deleted_count(result: Any) -> int:
    status = getattr(result, "status", None)
    if isinstance(status, dict):
        return int(status.get("deleted", 0))
    return 0


def _create_qdrant_client(url: str):
    try:
        from qdrant_client import QdrantClient
    except ImportError as error:
        message = (
            "Package `qdrant-client` is required for Qdrant ingestion. "
            "Install it with `pip install qdrant-client`."
        )
        raise ImportError(message) from error

    return QdrantClient(url=url)
