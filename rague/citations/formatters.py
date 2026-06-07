"""Markdown formatters for citation-backed answers."""

from __future__ import annotations

from collections.abc import Sequence

from rague.citations.models import CitationSource, CitedAnswer, CitedClaim

SOURCES_HEADING = "## Источники"


def format_sources_markdown(
    sources: Sequence[CitationSource],
    *,
    include_chunk_ids: bool = True,
    include_updated_at: bool = False,
) -> str:
    """Render a Markdown sources section from deduplicated citation targets."""
    if not sources:
        return f"{SOURCES_HEADING}\n\n_Источники не найдены._"

    lines = [SOURCES_HEADING, ""]
    for index, source in enumerate(sources, start=1):
        label = source.label or f"[{index}]"
        source_label = _format_source_label(source)
        lines.append(f"{label} {source_label}")
        if include_updated_at and source.source_updated_at:
            lines.append(f"   updated: {source.source_updated_at}")
        if include_chunk_ids and source.chunk_ids:
            chunk_preview = ", ".join(source.chunk_ids[:5])
            if len(source.chunk_ids) > 5:
                chunk_preview += f", ... (+{len(source.chunk_ids) - 5})"
            lines.append(f"   chunks: {chunk_preview}")

    return "\n".join(lines)


def format_answer_with_sources(
    answer: str,
    sources: Sequence[CitationSource],
    *,
    include_chunk_ids: bool = True,
) -> str:
    """Combine answer text with a Markdown sources section."""
    answer_text = answer.strip()
    sources_text = format_sources_markdown(
        sources,
        include_chunk_ids=include_chunk_ids,
    )
    if not answer_text:
        return sources_text
    return f"{answer_text}\n\n{sources_text}"


def format_claim_markdown(
    claim: CitedClaim,
    *,
    debug_chunk_ids: bool = False,
) -> str:
    """Render one cited claim with citation markers."""
    text = claim.text.strip()
    if not claim.citation_refs:
        return text

    markers: list[str] = []
    seen: set[str] = set()
    for ref in claim.citation_refs:
        marker = f"{ref.label}:{ref.chunk_id}" if debug_chunk_ids else ref.label
        if marker not in seen:
            markers.append(marker)
            seen.add(marker)

    return f"{text} {' '.join(markers)}"


def format_cited_answer_markdown(
    answer: CitedAnswer,
    *,
    debug_chunk_ids: bool = False,
    include_chunk_ids: bool = True,
    include_updated_at: bool = True,
) -> str:
    """Render structured cited answer with claim markers and sources section."""
    sections: list[str] = []

    if answer.intro:
        sections.append(answer.intro.strip())

    for claim in answer.claims:
        sections.append(
            format_claim_markdown(claim, debug_chunk_ids=debug_chunk_ids)
        )

    if answer.summary:
        sections.append(answer.summary.strip())

    body = "\n\n".join(section for section in sections if section)
    sources_text = format_sources_markdown(
        answer.sources,
        include_chunk_ids=include_chunk_ids,
        include_updated_at=include_updated_at,
    )

    if not body:
        return sources_text
    return f"{body}\n\n{sources_text}"


def _format_source_label(source: CitationSource) -> str:
    title = source.title or "unknown"
    path_suffix = f" — {source.path}" if source.path else ""

    if source.source:
        return f"[{title}]({source.source}){path_suffix}"

    if source.path:
        return f"{title}{path_suffix}"

    return title
