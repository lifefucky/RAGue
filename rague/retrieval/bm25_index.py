"""BM25 lexical retrieval over Qdrant chunk documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from rague.vectorstores.qdrant_store import QdrantChunkStore, ScoredDocument

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[\w\d_]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize chunk or query text for BM25 scoring."""
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


@dataclass
class Bm25ChunkIndex:
    """In-memory BM25 index built from Qdrant chunk payloads."""

    documents: list[Document] = field(default_factory=list)
    _bm25: BM25Okapi | None = None
    built_at: datetime | None = None

    @property
    def corpus_size(self) -> int:
        return len(self.documents)

    @classmethod
    def from_store(
        cls,
        store: QdrantChunkStore,
        *,
        filter_: Any | None = None,
    ) -> Bm25ChunkIndex:
        """Build a BM25 index by scrolling chunk documents from Qdrant."""
        documents = store.scroll_chunks(filter_=filter_)
        return cls.from_documents(documents)

    @classmethod
    def from_documents(cls, documents: list[Document]) -> Bm25ChunkIndex:
        """Build a BM25 index from in-memory chunk documents."""
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [tokenize(document.page_content) for document in documents]
        if not tokenized_corpus:
            return cls(documents=[], _bm25=None, built_at=_utc_now())

        return cls(
            documents=list(documents),
            _bm25=BM25Okapi(tokenized_corpus),
            built_at=_utc_now(),
        )

    def refresh_from_store(
        self,
        store: QdrantChunkStore,
        *,
        filter_: Any | None = None,
    ) -> Bm25ChunkIndex:
        """Rebuild this index from the latest Qdrant chunk corpus."""
        refreshed = self.from_store(store, filter_=filter_)
        self.documents = refreshed.documents
        self._bm25 = refreshed._bm25
        self.built_at = refreshed.built_at
        return self

    def search(self, query: str, *, limit: int = 50) -> list[ScoredDocument]:
        if not self.documents or self._bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[ScoredDocument] = []
        for index in ranked_indices[:limit]:
            score = float(scores[index])
            if score <= 0:
                continue
            results.append(
                ScoredDocument(
                    document=self.documents[index],
                    score=score,
                )
            )
        return results


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
