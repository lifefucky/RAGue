"""Embedding backend protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TextEmbedder(Protocol):
    """Embed document texts and retrieval queries into dense vectors."""

    model_name: str
    vector_size: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""

    def embed_query(self, text: str) -> list[float]:
        """Return one embedding vector for a retrieval query."""
