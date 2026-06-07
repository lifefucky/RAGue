"""Hybrid retrieval and reranking components."""

from rague.retrieval.bm25_index import Bm25ChunkIndex, tokenize
from rague.retrieval.cross_encoder_reranker import (
    DEFAULT_RERANKER_MODEL,
    RERANKER_MODEL_PRESETS,
    CrossEncoderReranker,
    create_reranker_from_env,
    resolve_reranker_model,
)
from rague.retrieval.hybrid_reranker import (
    HybridRerankerRetriever,
    HybridRetrieverConfig,
    build_retrieval_metadata_filter,
    create_retriever_from_config,
    create_retriever_from_env,
    merge_and_dedup_candidates,
)

__all__ = [
    "Bm25ChunkIndex",
    "CrossEncoderReranker",
    "DEFAULT_RERANKER_MODEL",
    "HybridRerankerRetriever",
    "HybridRetrieverConfig",
    "RERANKER_MODEL_PRESETS",
    "build_retrieval_metadata_filter",
    "create_reranker_from_env",
    "create_retriever_from_config",
    "create_retriever_from_env",
    "merge_and_dedup_candidates",
    "resolve_reranker_model",
    "tokenize",
]
