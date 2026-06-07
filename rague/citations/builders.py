"""Build deduplicated citation sources and context from retrieved documents."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document

from rague.citations.adapters import citation_target_for_metadata
from rague.citations.models import (
    CitationContext,
    CitationRef,
    CitationSource,
    CitationWarning,
    missing_fields_for_document,
)


def build_citation_sources(
    documents: Sequence[Document],
) -> tuple[list[CitationSource], list[CitationWarning]]:
    """Extract citation sources from retrieved documents.

    Deduplicates by ``document_id`` + ``source`` (fallback: ``source`` alone,
    then ``chunk_id``). Type-specific metadata fields are preserved via
    pass-through on each ``CitationSource.metadata``.
    """
    grouped: dict[str, dict[str, Any]] = {}
    warnings: list[CitationWarning] = []
    order: list[str] = []

    for document in documents:
        metadata = dict(document.metadata)
        chunk_id = _as_str(metadata.get("chunk_id")) or _as_str(document.id)
        missing = missing_fields_for_document(metadata)
        _append_missing_warnings(warnings, chunk_id, missing)

        key = _dedup_key(metadata, chunk_id)
        if key not in grouped:
            enriched_metadata = dict(metadata)
            enriched_metadata["citation_target"] = citation_target_for_metadata(metadata)
            grouped[key] = {
                "document_type": _as_str(metadata.get("document_type")) or "unknown",
                "document_id": _as_str(metadata.get("document_id")),
                "title": _as_str(metadata.get("title")) or "unknown",
                "source": _as_str(metadata.get("source")),
                "path": _as_str(metadata.get("path")),
                "source_updated_at": _as_str(metadata.get("source_updated_at")),
                "ingested_at": _as_str(metadata.get("ingested_at")),
                "metadata": enriched_metadata,
                "chunk_ids": [],
            }
            order.append(key)

        if chunk_id and chunk_id not in grouped[key]["chunk_ids"]:
            grouped[key]["chunk_ids"].append(chunk_id)

    sources: list[CitationSource] = []
    for index, key in enumerate(order, start=1):
        entry = grouped[key]
        document_id = entry["document_id"]
        source_url = entry["source"]
        source_id = document_id or source_url or key

        sources.append(
            CitationSource(
                source_id=source_id,
                document_type=entry["document_type"],
                document_id=document_id,
                chunk_ids=list(entry["chunk_ids"]),
                title=entry["title"],
                source=source_url,
                path=entry["path"],
                source_updated_at=entry["source_updated_at"],
                ingested_at=entry["ingested_at"],
                label=f"[{index}]",
                metadata=dict(entry["metadata"]),
            )
        )

    return sources, warnings


def build_citation_context(documents: Sequence[Document]) -> CitationContext:
    """Prepare citation context with stable source labels and chunk refs."""
    document_list = list(documents)
    sources, warnings = build_citation_sources(document_list)
    refs_by_chunk_id = _build_refs_by_chunk_id(document_list, sources)
    return CitationContext(
        documents=document_list,
        sources=sources,
        refs_by_chunk_id=refs_by_chunk_id,
        warnings=list(warnings),
    )


def _build_refs_by_chunk_id(
    documents: Sequence[Document],
    sources: Sequence[CitationSource],
) -> dict[str, CitationRef]:
    source_by_chunk_id: dict[str, CitationSource] = {}
    for source in sources:
        for chunk_id in source.chunk_ids:
            source_by_chunk_id[chunk_id] = source

    refs: dict[str, CitationRef] = {}
    for document in documents:
        metadata = dict(document.metadata)
        chunk_id = _as_str(metadata.get("chunk_id")) or _as_str(document.id)
        if not chunk_id or chunk_id in refs:
            continue

        source = source_by_chunk_id.get(chunk_id)
        label = source.label if source and source.label else "[?]"
        source_id = source.source_id if source else _as_str(metadata.get("document_id")) or chunk_id

        ref_metadata = dict(metadata)
        ref_metadata["citation_target"] = citation_target_for_metadata(metadata)

        refs[chunk_id] = CitationRef(
            source_id=source_id,
            chunk_id=chunk_id,
            label=label,
            document_id=_as_str(metadata.get("document_id")),
            source=_as_str(metadata.get("source")),
            metadata=ref_metadata,
        )

    return refs


def _append_missing_warnings(
    warnings: list[CitationWarning],
    chunk_id: str | None,
    missing: list[str],
) -> None:
    if not missing:
        return

    if "source" in missing:
        warnings.append(
            CitationWarning(
                chunk_id=chunk_id,
                missing_fields=tuple(missing),
                message=(
                    f"Chunk `{chunk_id or 'unknown'}` is missing citation "
                    f"field `source`; source will be listed without a link."
                ),
            )
        )
        return

    warnings.append(
        CitationWarning(
            chunk_id=chunk_id,
            missing_fields=tuple(missing),
            message=(
                f"Chunk `{chunk_id or 'unknown'}` is missing citation "
                f"fields: {', '.join(missing)}."
            ),
        )
    )


def _dedup_key(metadata: dict[str, Any], chunk_id: str | None) -> str:
    document_id = _as_str(metadata.get("document_id"))
    source = _as_str(metadata.get("source"))
    if document_id and source:
        return f"document:{document_id}:source:{source}"
    if document_id:
        return f"document:{document_id}"
    if source:
        return f"source:{source}"
    return f"chunk:{chunk_id or 'unknown'}"


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
