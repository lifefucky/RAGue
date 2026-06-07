from __future__ import annotations

import pytest

from rague.agents import llm


def test_create_chat_model_from_env_passes_env_values(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("RAGUE_CHAT_MODEL", "gpt-test")
    monkeypatch.setenv("RAGUE_CHAT_TEMPERATURE", "0.1")
    monkeypatch.setenv("RAGUE_CHAT_TIMEOUT", "30")
    monkeypatch.setenv("RAGUE_CHAT_MAX_RETRIES", "3")
    monkeypatch.setenv("RAGUE_AGENT_STREAMING", "1")

    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm, "ChatOpenAI", FakeChatOpenAI, raising=False)
    monkeypatch.setitem(llm.__dict__, "ChatOpenAI", FakeChatOpenAI)

    try:
        from langchain_openai import ChatOpenAI as _RealChatOpenAI
    except ImportError:
        _RealChatOpenAI = None

    monkeypatch.setattr(
        "rague.agents.llm.ChatOpenAI",
        FakeChatOpenAI,
        raising=False,
    )

    # Patch inside function via import
    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)

    model = llm.create_chat_model_from_env()

    assert model is not None
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["model"] == "gpt-test"
    assert captured["temperature"] == 0.1
    assert captured["timeout"] == 30.0
    assert captured["max_retries"] == 3
    assert captured["streaming"] is True


def test_create_chat_model_from_env_missing_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            del kwargs

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        llm.create_chat_model_from_env()
