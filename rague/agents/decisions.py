"""Production LLM decision adapter for agentic RAG workflow."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

if TYPE_CHECKING:
    from rague.evaluation.tracing import AgentDecisionObserver

from rague.agents.parsers import (
    DocumentRelevanceOutput,
    GeneratedAnswerOutput,
    RewriteQueryOutput,
    ShouldRetrieveOutput,
    generated_output_to_generated_answer,
    parse_json_output,
)
from rague.agents.prompts import build_chat_prompt, format_documents_context, load_prompt_config
from rague.agents.workflows import GeneratedAnswer, RelevanceDecision
from rague.citations.models import CitationContext


class AgentLlmDecisions:
    """Map YAML prompts + chat model outputs to workflow decision callables."""

    def __init__(
        self,
        chat_model: Any,
        *,
        prompt_version_overrides: dict[str, str] | None = None,
        max_context_chars_per_doc: int = 1200,
        use_structured_output: bool = True,
        observer: AgentDecisionObserver | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.prompt_version_overrides = prompt_version_overrides or {}
        self.max_context_chars_per_doc = max_context_chars_per_doc
        self.use_structured_output = use_structured_output
        self.observer = observer

    def decide_should_retrieve(self, question: str) -> bool:
        output = self._invoke_structured(
            task_name="should_retrieve",
            output_model=ShouldRetrieveOutput,
            variables={"question": question},
        )
        return bool(output.needs_retrieval)

    def grade_documents(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> RelevanceDecision:
        output = self._invoke_structured(
            task_name="grade_documents",
            output_model=DocumentRelevanceOutput,
            variables={
                "query": query,
                "documents_context": format_documents_context(
                    documents,
                    max_chars_per_doc=self.max_context_chars_per_doc,
                ),
            },
        )
        return RelevanceDecision(
            is_relevant=bool(output.is_relevant),
            reason=output.reason or None,
        )

    def rewrite_query(
        self,
        question: str,
        query: str,
        documents: Sequence[Document],
    ) -> str:
        output = self._invoke_structured(
            task_name="rewrite_query",
            output_model=RewriteQueryOutput,
            variables={
                "question": question,
                "query": query,
                "documents_context": format_documents_context(
                    documents,
                    max_chars_per_doc=self.max_context_chars_per_doc,
                ),
            },
        )
        rewritten = output.query.strip()
        return rewritten or query

    def generate_answer(
        self,
        question: str,
        documents: Sequence[Document],
        citation_context: CitationContext | None,
    ) -> GeneratedAnswer:
        allowed_chunk_ids: list[str] = []
        if citation_context is not None:
            allowed_chunk_ids = sorted(citation_context.refs_by_chunk_id.keys())

        if not documents:
            output = self._invoke_structured(
                task_name="generate_answer",
                output_model=GeneratedAnswerOutput,
                variables={
                    "question": question,
                    "documents_context": "_Документы не найдены._",
                    "allowed_chunk_ids": "[]",
                },
            )
            return generated_output_to_generated_answer(output)

        output = self._invoke_structured(
            task_name="generate_answer",
            output_model=GeneratedAnswerOutput,
            variables={
                "question": question,
                "documents_context": format_documents_context(
                    documents,
                    max_chars_per_doc=self.max_context_chars_per_doc,
                ),
                "allowed_chunk_ids": ", ".join(allowed_chunk_ids) or "[]",
            },
        )
        return generated_output_to_generated_answer(
            output,
            allowed_chunk_ids=allowed_chunk_ids,
        )

    def _prompt_version(self, task_name: str) -> str | None:
        return self.prompt_version_overrides.get(task_name)

    def _invoke_structured(
        self,
        *,
        task_name: str,
        output_model: type[Any],
        variables: dict[str, Any],
    ) -> Any:
        prompt = build_chat_prompt(task_name, version=self._prompt_version(task_name))
        messages = prompt.format_messages(**variables)

        if self.use_structured_output:
            try:
                structured_model = self.chat_model.with_structured_output(output_model)
                result = structured_model.invoke(messages)
                if isinstance(result, output_model):
                    output = result
                else:
                    output = output_model.model_validate(result)
                self._notify_observer(
                    task_name=task_name,
                    variables=variables,
                    output=output,
                )
                return output
            except Exception:
                pass

        response = self.chat_model.invoke(messages)
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            content = str(content)
        output = parse_json_output(content, output_model)
        self._notify_observer(task_name=task_name, variables=variables, output=output)
        return output

    def _notify_observer(
        self,
        *,
        task_name: str,
        variables: dict[str, Any],
        output: Any,
    ) -> None:
        if self.observer is None:
            return
        self.observer.on_task_output(task_name, variables, output)

    def task_temperature(self, task_name: str) -> float | None:
        """Return YAML-configured temperature for a task, if present."""
        version = load_prompt_config(task_name, version=self._prompt_version(task_name))
        return version.temperature
