from rague.embeddings.factory import _is_e5_model, _prefix_e5_passages


def test_is_e5_model_detects_multilingual_e5() -> None:
    assert _is_e5_model("intfloat/multilingual-e5-base") is True
    assert _is_e5_model("sentence-transformers/all-MiniLM-L6-v2") is False


def test_prefix_e5_passages_adds_prefix() -> None:
    assert _prefix_e5_passages(["текст"]) == ["passage: текст"]


def test_prefix_e5_passages_does_not_duplicate() -> None:
    assert _prefix_e5_passages(["passage: уже есть"]) == ["passage: уже есть"]
    assert _prefix_e5_passages(["Passage: mixed case"]) == ["Passage: mixed case"]
