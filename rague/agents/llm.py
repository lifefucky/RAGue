"""OpenAI-compatible chat model factory for agentic RAG."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_CHAT_TEMPERATURE = 0.0
DEFAULT_CHAT_TIMEOUT = 60.0
DEFAULT_CHAT_MAX_RETRIES = 2


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def create_chat_model_from_env():
    """Create an OpenAI-compatible chat model from environment variables."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise ImportError(
            "Agentic RAG chat backend requires `langchain-openai`. "
            "Install project requirements or provide a custom chat model."
        ) from error

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "OPENAI_API_KEY is required for production agent workflow. "
            "Set it in `.env` or pass a custom chat model to the workflow."
        )

    kwargs: dict[str, Any] = {
        "model": os.getenv("RAGUE_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        "temperature": _env_float("RAGUE_CHAT_TEMPERATURE", DEFAULT_CHAT_TEMPERATURE),
        "timeout": _env_float("RAGUE_CHAT_TIMEOUT", DEFAULT_CHAT_TIMEOUT),
        "max_retries": _env_int("RAGUE_CHAT_MAX_RETRIES", DEFAULT_CHAT_MAX_RETRIES),
        "api_key": api_key,
        "streaming": _env_bool("RAGUE_AGENT_STREAMING", default=False),
    }

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url and base_url.strip():
        kwargs["base_url"] = base_url.strip()

    return ChatOpenAI(**kwargs)
