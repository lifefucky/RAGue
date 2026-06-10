from __future__ import annotations

import pytest

from rague.agents.parsers import (
    ClaimOutput,
    GeneratedAnswerOutput,
    ShouldRetrieveOutput,
    filter_claim_chunk_ids,
    generated_output_to_generated_answer,
    parse_json_output,
)


def test_parse_json_output_valid_payload() -> None:
    parsed = parse_json_output(
        '{"needs_retrieval": true, "reason": "needs docs"}',
        ShouldRetrieveOutput,
    )

    assert parsed.needs_retrieval is True
    assert parsed.reason == "needs docs"


def test_parse_json_output_invalid_json_raises() -> None:
    with pytest.raises(ValueError, match="Failed to parse JSON output"):
        parse_json_output("not-json", ShouldRetrieveOutput)


def test_filter_claim_chunk_ids_removes_unknown_ids() -> None:
    claims = [
        ClaimOutput(text="Claim A", chunk_ids=["chunk-1", "missing"]),
        ClaimOutput(text="Claim B", chunk_ids=["chunk-2"]),
    ]

    filtered = filter_claim_chunk_ids(claims, {"chunk-1", "chunk-2"})

    assert len(filtered) == 2
    assert filtered[0].chunk_ids == ["chunk-1"]
    assert filtered[1].chunk_ids == ["chunk-2"]


def test_generated_output_to_generated_answer_with_claim_specs() -> None:
    output = GeneratedAnswerOutput(
        claims=[ClaimOutput(text="Answer claim.", chunk_ids=["chunk-1"])],
        intro="Intro",
        summary="Summary",
    )

    generated = generated_output_to_generated_answer(
        output,
        allowed_chunk_ids={"chunk-1"},
    )

    assert generated.claim_specs == [("Answer claim.", ["chunk-1"])]
    assert generated.intro == "Intro"
    assert generated.summary == "Summary"


def test_generated_output_to_generated_answer_preserves_answer_text_with_claims() -> None:
    output = GeneratedAnswerOutput(
        answer_text="Full cohesive answer for the user.",
        claims=[ClaimOutput(text="Supporting fact.", chunk_ids=["chunk-1"])],
    )

    generated = generated_output_to_generated_answer(
        output,
        allowed_chunk_ids={"chunk-1"},
    )

    assert generated.answer_text == "Full cohesive answer for the user."
    assert generated.claim_specs == [("Supporting fact.", ["chunk-1"])]
