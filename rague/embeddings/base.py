"""Embedding backend protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TextEmbedder(Protocol):
    """Embed document texts into dense vectors."""

    model_name: str
    vector_size: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
