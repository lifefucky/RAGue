"""Factory for pluggable embedding backends."""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any

from rague.embeddings.base import TextEmbedder


class DeterministicHashEmbedder:
    """Local fallback embedder for development and smoke tests."""

    def __init__(self, *, model_name: str = "deterministic-hash", vector_size: int = 384):
        self.model_name = model_name
        self.vector_size = vector_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hash_to_vector(text, self.vector_size) for text in texts]


def create_embedder(
    *,
    provider: str | None = None,
    model_name: str | None = None,
    vector_size: int | None = None,
    **kwargs: Any,
) -> TextEmbedder:
    """Create an embedder from explicit args or environment variables."""
    selected_provider = (
        provider or os.getenv("EMBEDDING_PROVIDER", "deterministic").strip().lower()
    )
    selected_model = model_name or os.getenv("EMBEDDING_MODEL")
    selected_vector_size = vector_size or int(os.getenv("EMBEDDING_VECTOR_SIZE", "384"))

    if selected_provider in {"deterministic", "hash", "fake"}:
        return DeterministicHashEmbedder(
            model_name=selected_model or "deterministic-hash",
            vector_size=selected_vector_size,
        )

    if selected_provider in {"openai", "openai_compatible"}:
        return _create_openai_compatible_embedder(
            model_name=selected_model or "text-embedding-3-small",
            vector_size=selected_vector_size,
            **kwargs,
        )

    if selected_provider in {"sentence_transformers", "local"}:
        return _create_sentence_transformers_embedder(
            model_name=selected_model or "sentence-transformers/all-MiniLM-L6-v2",
            vector_size=selected_vector_size,
        )

    message = (
        f"Unsupported embedding provider `{selected_provider}`. "
        "Use one of: deterministic, openai_compatible, sentence_transformers."
    )
    raise ValueError(message)


def _create_openai_compatible_embedder(
    *,
    model_name: str,
    vector_size: int,
    **kwargs: Any,
) -> TextEmbedder:
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as error:
        message = (
            "Provider `openai_compatible` requires `langchain-openai`. "
            "Install it or use `EMBEDDING_PROVIDER=deterministic`."
        )
        raise ImportError(message) from error

    api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
    base_url = kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")

    embedder = OpenAIEmbeddings(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )

    class OpenAICompatibleEmbedder:
        def __init__(self, inner: OpenAIEmbeddings, name: str, size: int) -> None:
            self._inner = inner
            self.model_name = name
            self.vector_size = size

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._inner.embed_documents(texts)

    return OpenAICompatibleEmbedder(embedder, model_name, vector_size)


def _create_sentence_transformers_embedder(
    *,
    model_name: str,
    vector_size: int,
) -> TextEmbedder:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as error:
        message = (
            "Provider `sentence_transformers` requires `langchain-community` "
            "with sentence-transformers installed."
        )
        raise ImportError(message) from error

    embedder = HuggingFaceEmbeddings(model_name=model_name)

    class SentenceTransformersEmbedder:
        def __init__(self, inner: HuggingFaceEmbeddings, name: str, size: int) -> None:
            self._inner = inner
            self.model_name = name
            self.vector_size = size

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._inner.embed_documents(texts)

    return SentenceTransformersEmbedder(embedder, model_name, vector_size)


def _hash_to_vector(text: str, vector_size: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    counter = 0

    while len(values) < vector_size:
        block = hashlib.sha256(digest + counter.to_bytes(4, "big")).digest()
        for byte in block:
            values.append((byte / 255.0) * 2.0 - 1.0)
            if len(values) >= vector_size:
                break
        counter += 1

    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]
