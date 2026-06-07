"""Helpers for building structured cited answers from citation context."""

from __future__ import annotations

from collections.abc import Sequence

from rague.citations.models import (
    CitationContext,
    CitationWarning,
    CitedAnswer,
    CitedClaim,
)


def cite_claim(
    text: str,
    chunk_ids: Sequence[str],
    context: CitationContext,
) -> tuple[CitedClaim, list[CitationWarning]]:
    """Link one answer claim to retrieved chunk ids."""
    refs = []
    warnings: list[CitationWarning] = []

    for chunk_id in chunk_ids:
        ref = context.refs_by_chunk_id.get(chunk_id)
        if ref is None:
            warnings.append(
                CitationWarning(
                    chunk_id=chunk_id,
                    missing_fields=(),
                    message=(
                        f"Claim references unknown chunk_id `{chunk_id}`; "
                        "citation marker will be omitted for this chunk."
                    ),
                )
            )
            continue
        if ref not in refs:
            refs.append(ref)

    return CitedClaim(text=text.strip(), citation_refs=refs), warnings


def build_cited_answer(
    claims: Sequence[str | CitedClaim],
    context: CitationContext,
    *,
    intro: str | None = None,
    summary: str | None = None,
) -> CitedAnswer:
    """Assemble a structured cited answer from claims and citation context."""
    built_claims: list[CitedClaim] = []
    warnings = list(context.warnings)

    for claim in claims:
        if isinstance(claim, CitedClaim):
            built_claims.append(claim)
            continue

        cited_claim, claim_warnings = cite_claim(claim, [], context)
        built_claims.append(cited_claim)
        warnings.extend(claim_warnings)

    return CitedAnswer(
        claims=built_claims,
        sources=list(context.sources),
        warnings=warnings,
        intro=intro,
        summary=summary,
    )


def build_cited_answer_from_claim_specs(
    claim_specs: Sequence[tuple[str, Sequence[str]]],
    context: CitationContext,
    *,
    intro: str | None = None,
    summary: str | None = None,
) -> CitedAnswer:
    """Build cited answer from (claim_text, chunk_ids) pairs."""
    built_claims: list[CitedClaim] = []
    warnings = list(context.warnings)

    for text, chunk_ids in claim_specs:
        cited_claim, claim_warnings = cite_claim(text, chunk_ids, context)
        built_claims.append(cited_claim)
        warnings.extend(claim_warnings)

    return CitedAnswer(
        claims=built_claims,
        sources=list(context.sources),
        warnings=warnings,
        intro=intro,
        summary=summary,
    )
