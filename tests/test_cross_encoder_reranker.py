from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rague.retrieval.cross_encoder_reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    create_reranker_from_env,
    resolve_reranker_model,
)


def test_resolve_reranker_model_supports_presets() -> None:
    assert resolve_reranker_model("bge_m3") == "BAAI/bge-reranker-v2-m3"
    assert (
        resolve_reranker_model("ms_marco")
        == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    assert (
        resolve_reranker_model("BAAI/bge-reranker-v2-m3")
        == "BAAI/bge-reranker-v2-m3"
    )


def test_score_pairs_delegates_to_cross_encoder_predict() -> None:
    reranker = CrossEncoderReranker("bge_m3")
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [0.8, 0.3]
    reranker._encoder = mock_encoder

    pairs = [("query", "doc one"), ("query", "doc two")]
    scores = reranker.score_pairs(pairs)

    mock_encoder.predict.assert_called_once_with(pairs)
    assert mock_encoder.predict.call_args.kwargs == {}
    assert scores == [0.8, 0.3]


def test_score_pairs_passes_batch_size_when_configured() -> None:
    reranker = CrossEncoderReranker("bge_m3", batch_size=8)
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [0.8]
    reranker._encoder = mock_encoder

    pairs = [("query", "doc one")]
    scores = reranker.score_pairs(pairs)

    mock_encoder.predict.assert_called_once_with(pairs, batch_size=8)
    assert scores == [0.8]


def test_score_pairs_returns_empty_for_empty_input() -> None:
    reranker = CrossEncoderReranker("bge_m3")
    reranker._encoder = MagicMock()

    assert reranker.score_pairs([]) == []


def test_rerank_returns_indices_sorted_by_score() -> None:
    reranker = CrossEncoderReranker("bge_m3")
    reranker._encoder = MagicMock()
    reranker._encoder.predict.return_value = [0.2, 0.9, 0.5]

    ranked = reranker.rerank("query", ["a", "b", "c"])

    assert ranked == [(1, 0.9), (2, 0.5), (0, 0.2)]


def test_rerank_returns_empty_for_empty_documents() -> None:
    reranker = CrossEncoderReranker("bge_m3")

    assert reranker.rerank("query", []) == []


def test_create_reranker_from_env_resolves_preset(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "ms_marco")

    reranker = create_reranker_from_env()

    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_create_reranker_from_env_uses_default(monkeypatch) -> None:
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    monkeypatch.delenv("RERANKER_BATCH_SIZE", raising=False)

    reranker = create_reranker_from_env()

    assert reranker.model_name == DEFAULT_RERANKER_MODEL
    assert reranker.batch_size is None


def test_create_reranker_from_env_reads_batch_size(monkeypatch) -> None:
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    monkeypatch.setenv("RERANKER_BATCH_SIZE", "16")

    reranker = create_reranker_from_env()

    assert reranker.batch_size == 16


def test_lazy_encoder_import() -> None:
    pytest.importorskip("sentence_transformers")
    reranker = CrossEncoderReranker("bge_m3")

    with patch(
        "sentence_transformers.CrossEncoder",
        return_value=MagicMock(),
    ) as mock_cross_encoder:
        encoder = reranker._get_encoder()

    mock_cross_encoder.assert_called_once_with("BAAI/bge-reranker-v2-m3")
    assert encoder is reranker._encoder
