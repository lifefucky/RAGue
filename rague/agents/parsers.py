"""Structured output schemas and JSON parsers for agentic RAG."""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, Field

from rague.agents.workflows import GeneratedAnswer

TModel = TypeVar("TModel", bound=BaseModel)


class ShouldRetrieveOutput(BaseModel):
    needs_retrieval: bool
    reason: str = ""


class DocumentRelevanceOutput(BaseModel):
    is_relevant: bool
    reason: str = ""


class RewriteQueryOutput(BaseModel):
    query: str
    reason: str = ""


class ClaimOutput(BaseModel):
    text: str
    chunk_ids: list[str] = Field(default_factory=list)


class GeneratedAnswerOutput(BaseModel):
    answer_text: str | None = None
    claims: list[ClaimOutput] = Field(default_factory=list)
    intro: str | None = None
    summary: str | None = None


def _extract_json_payload(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise ValueError("LLM response is empty.")

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]

    return stripped


def parse_json_output(text: str, model: type[TModel]) -> TModel:
    """Parse JSON from plain LLM text into a Pydantic model."""
    payload = _extract_json_payload(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"Failed to parse JSON output: {error}") from error

    return model.model_validate(data)


def filter_claim_chunk_ids(
    claims: list[ClaimOutput],
    allowed_chunk_ids: set[str] | list[str],
) -> list[ClaimOutput]:
    """Drop unknown chunk ids before citation assembly."""
    allowed = set(allowed_chunk_ids)
    filtered: list[ClaimOutput] = []
    for claim in claims:
        valid_ids = [chunk_id for chunk_id in claim.chunk_ids if chunk_id in allowed]
        if not claim.text.strip():
            continue
        filtered.append(ClaimOutput(text=claim.text.strip(), chunk_ids=valid_ids))
    return filtered


def generated_output_to_generated_answer(
    output: GeneratedAnswerOutput,
    *,
    allowed_chunk_ids: set[str] | list[str] | None = None,
) -> GeneratedAnswer:
    """Convert structured LLM output into workflow `GeneratedAnswer`."""
    claims = output.claims
    if allowed_chunk_ids is not None:
        claims = filter_claim_chunk_ids(claims, allowed_chunk_ids)

    claim_specs = [(claim.text, list(claim.chunk_ids)) for claim in claims if claim.text]
    if claim_specs:
        return GeneratedAnswer(
            claim_specs=claim_specs,
            intro=output.intro,
            summary=output.summary,
        )

    answer_text = (output.answer_text or "").strip()
    if answer_text:
        return GeneratedAnswer(
            answer_text=answer_text,
            intro=output.intro,
            summary=output.summary,
        )

    return GeneratedAnswer(intro=output.intro, summary=output.summary)
