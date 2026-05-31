"""Embedding backends for ingestion pipelines."""

from rague.embeddings.base import TextEmbedder
from rague.embeddings.factory import create_embedder

__all__ = ["TextEmbedder", "create_embedder"]
