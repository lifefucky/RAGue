"""Type-specific citation target adapters with permissive fallback."""

from __future__ import annotations

from typing import Any


def citation_target_for_page(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build citation target for Confluence page chunks."""
    return {
        "target_type": "page",
        "source": metadata.get("source"),
        "title": metadata.get("title"),
        "path": metadata.get("path"),
        "page_id": metadata.get("page_id"),
        "document_id": metadata.get("document_id"),
    }


def citation_target_for_code(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build citation target for code-related chunks (inherits page URL)."""
    target = citation_target_for_page(metadata)
    target["target_type"] = "code"
    for key in ("code_ref", "code_language", "caption", "chunk_type", "local_heading"):
        if metadata.get(key) is not None:
            target[key] = metadata[key]
    return target


def citation_target_for_attachment(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build future-compatible citation target for attachment chunks."""
    return {
        "target_type": "attachment",
        "source": metadata.get("source"),
        "title": metadata.get("title") or metadata.get("attachment_filename"),
        "path": metadata.get("path"),
        "document_id": metadata.get("document_id"),
        "attachment_id": metadata.get("attachment_id"),
        "attachment_filename": metadata.get("attachment_filename"),
        "parent_page_id": metadata.get("parent_page_id"),
    }


def citation_target_for_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Select citation adapter by document/chunk type with permissive fallback."""
    document_type = metadata.get("document_type")
    chunk_type = metadata.get("chunk_type")

    if document_type == "attachment":
        return citation_target_for_attachment(metadata)
    if chunk_type in {"code", "code_summary"}:
        return citation_target_for_code(metadata)
    if document_type == "page":
        return citation_target_for_page(metadata)

    return {
        "target_type": document_type or chunk_type or "unknown",
        "source": metadata.get("source"),
        "title": metadata.get("title"),
        "path": metadata.get("path"),
        "document_id": metadata.get("document_id"),
    }
