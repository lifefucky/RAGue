from rague.embeddings.factory import (
    DeterministicHashEmbedder,
    _is_e5_model,
    _prefix_e5_passages,
    _prefix_e5_query,
)


def test_is_e5_model_detects_multilingual_e5() -> None:
    assert _is_e5_model("intfloat/multilingual-e5-base") is True
    assert _is_e5_model("sentence-transformers/all-MiniLM-L6-v2") is False


def test_prefix_e5_passages_adds_prefix() -> None:
    assert _prefix_e5_passages(["текст"]) == ["passage: текст"]


def test_prefix_e5_passages_does_not_duplicate() -> None:
    assert _prefix_e5_passages(["passage: уже есть"]) == ["passage: уже есть"]
    assert _prefix_e5_passages(["Passage: mixed case"]) == ["Passage: mixed case"]


def test_prefix_e5_query_adds_prefix() -> None:
    assert _prefix_e5_query("что такое RAG") == "query: что такое RAG"


def test_prefix_e5_query_does_not_duplicate() -> None:
    assert _prefix_e5_query("query: уже есть") == "query: уже есть"
    assert _prefix_e5_query("Query: mixed case") == "Query: mixed case"


def test_deterministic_embed_query_returns_vector() -> None:
    embedder = DeterministicHashEmbedder(vector_size=128)
    vector = embedder.embed_query("что такое Debezium")

    assert len(vector) == 128
    assert all(isinstance(value, float) for value in vector)
