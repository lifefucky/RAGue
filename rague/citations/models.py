"""Citation models with a minimal contract and flexible metadata pass-through."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

# Minimal fields required to build a verifiable citation target.
REQUIRED_CITATION_FIELDS = (
    "document_type",
    "document_id",
    "chunk_id",
    "source",
    "title",
)

# Required metadata for Confluence page chunks (step 2 contract).
REQUIRED_PAGE_CITATION_FIELDS = (
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
)


@dataclass(frozen=True)
class CitationWarning:
    """Non-fatal issue encountered while building citation sources."""

    chunk_id: str | None
    missing_fields: tuple[str, ...]
    message: str


@dataclass
class CitationSource:
    """Deduplicated citation target derived from one or more retrieved chunks."""

    source_id: str
    document_type: str
    document_id: str | None
    chunk_ids: list[str] = field(default_factory=list)
    title: str = "unknown"
    source: str | None = None
    path: str | None = None
    source_updated_at: str | None = None
    ingested_at: str | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CitationRef:
    """Reference to a specific retrieved chunk within a citation context."""

    source_id: str
    chunk_id: str
    label: str
    document_id: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CitationContext:
    """Prepared citation context from retrieved documents."""

    documents: list[Document] = field(default_factory=list)
    sources: list[CitationSource] = field(default_factory=list)
    refs_by_chunk_id: dict[str, CitationRef] = field(default_factory=dict)
    warnings: list[CitationWarning] = field(default_factory=list)


@dataclass
class CitedClaim:
    """One important answer statement linked to retrieved chunks."""

    text: str
    citation_refs: list[CitationRef] = field(default_factory=list)


@dataclass
class CitedAnswer:
    """Structured answer with claim-level citations and source list."""

    claims: list[CitedClaim] = field(default_factory=list)
    sources: list[CitationSource] = field(default_factory=list)
    warnings: list[CitationWarning] = field(default_factory=list)
    intro: str | None = None
    summary: str | None = None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def missing_citation_fields(metadata: dict[str, Any]) -> list[str]:
    """Return base citation fields absent from chunk metadata."""
    return [
        field_name
        for field_name in REQUIRED_CITATION_FIELDS
        if _is_missing(metadata.get(field_name))
    ]


def missing_page_citation_fields(metadata: dict[str, Any]) -> list[str]:
    """Return page-specific citation fields absent from chunk metadata."""
    return [
        field_name
        for field_name in REQUIRED_PAGE_CITATION_FIELDS
        if _is_missing(metadata.get(field_name))
    ]


def missing_fields_for_document(metadata: dict[str, Any]) -> list[str]:
    """Return required citation fields for a document based on its type."""
    document_type = metadata.get("document_type")
    if document_type == "page":
        return missing_page_citation_fields(metadata)
    return missing_citation_fields(metadata)
