"""YAML prompt loader and document context formatting for agentic RAG."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

AGENTIC_RAG_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "agentic_rag"


@dataclass(frozen=True)
class PromptVersion:
    """One versioned prompt definition loaded from YAML."""

    system: str
    user: str
    input_variables: list[str]
    few_shot: list[dict[str, Any]] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class PromptConfig:
    """Full prompt config for one agent task."""

    task_name: str
    current_version: str
    versions: dict[str, PromptVersion]


def _prompt_file_path(task_name: str) -> Path:
    return AGENTIC_RAG_PROMPT_DIR / f"{task_name}.yaml"


def _parse_prompt_version(raw: dict[str, Any]) -> PromptVersion:
    system = raw.get("system")
    user = raw.get("user")
    input_variables = raw.get("input_variables")
    if not isinstance(system, str) or not system.strip():
        raise ValueError("Prompt version must include non-empty `system`.")
    if not isinstance(user, str) or not user.strip():
        raise ValueError("Prompt version must include non-empty `user`.")
    if not isinstance(input_variables, list) or not input_variables:
        raise ValueError("Prompt version must include non-empty `input_variables`.")

    few_shot = raw.get("few_shot") or []
    if not isinstance(few_shot, list):
        raise ValueError("Prompt version field `few_shot` must be a list.")

    temperature = raw.get("temperature")
    max_tokens = raw.get("max_tokens")
    return PromptVersion(
        system=system.strip(),
        user=user.strip(),
        input_variables=[str(name) for name in input_variables],
        few_shot=list(few_shot),
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=int(max_tokens) if max_tokens is not None else None,
    )


def load_prompt_config(task_name: str, version: str | None = None) -> PromptVersion:
    """Load one prompt version from YAML by task name."""
    path = _prompt_file_path(task_name)
    if not path.exists():
        raise FileNotFoundError(f"Prompt config not found for task `{task_name}`: {path}")

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    current_version = raw.get("current_version")
    if not current_version:
        raise ValueError(f"Prompt config `{task_name}` is missing `current_version`.")

    versions_raw = raw.get("versions") or {}
    if not isinstance(versions_raw, dict) or not versions_raw:
        raise ValueError(f"Prompt config `{task_name}` is missing `versions`.")

    selected_version = version or str(current_version)
    version_raw = versions_raw.get(selected_version)
    if version_raw is None:
        raise ValueError(
            f"Prompt config `{task_name}` does not define version `{selected_version}`."
        )

    return _parse_prompt_version(version_raw)


def load_prompt_config_bundle(task_name: str) -> PromptConfig:
    """Load full prompt config including all versions."""
    path = _prompt_file_path(task_name)
    if not path.exists():
        raise FileNotFoundError(f"Prompt config not found for task `{task_name}`: {path}")

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    current_version = raw.get("current_version")
    if not current_version:
        raise ValueError(f"Prompt config `{task_name}` is missing `current_version`.")

    versions_raw = raw.get("versions") or {}
    if not isinstance(versions_raw, dict) or not versions_raw:
        raise ValueError(f"Prompt config `{task_name}` is missing `versions`.")

    versions = {
        str(name): _parse_prompt_version(version_raw)
        for name, version_raw in versions_raw.items()
    }
    return PromptConfig(
        task_name=task_name,
        current_version=str(current_version),
        versions=versions,
    )


def build_chat_prompt(task_name: str, version: str | None = None) -> ChatPromptTemplate:
    """Build a LangChain chat prompt from YAML config."""
    prompt_version = load_prompt_config(task_name, version=version)
    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt_version.system),
            ("human", prompt_version.user),
        ]
    )


def format_documents_context(
    documents: Sequence[Document],
    *,
    max_chars_per_doc: int = 1200,
) -> str:
    """Render retrieved documents into compact prompt context."""
    if not documents:
        return "_Документы не найдены._"

    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        chunk_id = metadata.get("chunk_id") or document.id or f"doc-{index}"
        title = metadata.get("title") or "unknown"
        path = metadata.get("path") or ""
        source = metadata.get("source") or ""
        rerank_score = metadata.get("rerank_score")
        score_text = f"{rerank_score:.4f}" if isinstance(rerank_score, (int, float)) else "n/a"

        content = document.page_content.strip()
        if len(content) > max_chars_per_doc:
            content = content[: max_chars_per_doc - 3].rstrip() + "..."

        blocks.append(
            "\n".join(
                [
                    f"[{index}] chunk_id={chunk_id}",
                    f"title={title}",
                    f"path={path}",
                    f"source={source}",
                    f"rerank_score={score_text}",
                    "content:",
                    content or "_empty_",
                ]
            )
        )

    return "\n\n".join(blocks)
