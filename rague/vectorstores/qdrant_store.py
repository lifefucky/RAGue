"""Qdrant vector store helpers for Confluence chunk ingestion."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.documents import Document

DEFAULT_COLLECTION = "confluence_pages_v1"
DEFAULT_DISTANCE = "Cosine"

PAYLOAD_INDEX_FIELDS = (
    "source_type",
    "document_type",
    "document_id",
    "chunk_id",
    "page_id",
    "space",
    "source_updated_at",
    "ingested_at",
    "parent_page_id",
    "is_current",
)


@dataclass
class UpsertStats:
    points_upserted: int = 0
    points_deleted: int = 0


class QdrantChunkStore:
    """Manage Confluence chunk points in a Qdrant collection."""

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        collection_name: str = DEFAULT_COLLECTION,
        vector_size: int,
        distance: str = DEFAULT_DISTANCE,
    ) -> None:
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self._client = _create_qdrant_client(url)

    def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        if not self._client.collection_exists(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=distance_map.get(self.distance, Distance.COSINE),
                ),
            )

        self._ensure_payload_indexes()

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


def _stable_point_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


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
