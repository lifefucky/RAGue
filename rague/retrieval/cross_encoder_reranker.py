"""Cross-encoder reranker wrapper for query-document scoring."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_BATCH_SIZE: int | None = None

RERANKER_MODEL_PRESETS: dict[str, str] = {
    "bge_m3": "BAAI/bge-reranker-v2-m3",
    "ms_marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


def resolve_reranker_model(model_name: str) -> str:
    """Resolve preset alias or return the full Hugging Face model name."""
    normalized = model_name.strip()
    return RERANKER_MODEL_PRESETS.get(normalized, normalized)


class CrossEncoderReranker:
    """Score query-document pairs with a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int | None = DEFAULT_RERANKER_BATCH_SIZE,
    ) -> None:
        self.model_name = resolve_reranker_model(model_name)
        self.batch_size = batch_size
        self._encoder: Any = None

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return relevance scores for each query-document pair."""
        if not pairs:
            return []
        predict_kwargs: dict[str, Any] = {}
        if self.batch_size is not None:
            predict_kwargs["batch_size"] = self.batch_size
        raw_scores = self._get_encoder().predict(pairs, **predict_kwargs)
        return [float(score) for score in raw_scores]

    def rerank(
        self,
        query: str,
        documents: list[str],
    ) -> list[tuple[int, float]]:
        """Return document indices and scores sorted by descending relevance."""
        if not documents:
            return []

        pairs = [(query, document) for document in documents]
        scores = self.score_pairs(pairs)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(index, float(score)) for index, score in ranked]

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import CrossEncoder

            self._encoder = CrossEncoder(self.model_name)
        return self._encoder


def create_reranker_from_env() -> CrossEncoderReranker:
    """Build a reranker from the RERANKER_MODEL environment variable."""
    model_name = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    batch_size_raw = os.getenv("RERANKER_BATCH_SIZE", "").strip()
    batch_size = int(batch_size_raw) if batch_size_raw else None
    return CrossEncoderReranker(model_name, batch_size=batch_size)
