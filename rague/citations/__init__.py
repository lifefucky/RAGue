"""Citation utilities for retrieved chunk metadata and answer transparency."""

from rague.citations.adapters import (
    citation_target_for_attachment,
    citation_target_for_code,
    citation_target_for_metadata,
    citation_target_for_page,
)
from rague.citations.answers import (
    build_cited_answer,
    build_cited_answer_from_claim_specs,
    cite_claim,
)
from rague.citations.builders import build_citation_context, build_citation_sources
from rague.citations.formatters import (
    SOURCES_HEADING,
    format_answer_with_sources,
    format_cited_answer_markdown,
    format_claim_markdown,
    format_sources_markdown,
)
from rague.citations.models import (
    REQUIRED_CITATION_FIELDS,
    REQUIRED_PAGE_CITATION_FIELDS,
    CitationContext,
    CitationRef,
    CitationSource,
    CitationWarning,
    CitedAnswer,
    CitedClaim,
    missing_citation_fields,
    missing_fields_for_document,
    missing_page_citation_fields,
)

__all__ = [
    "REQUIRED_CITATION_FIELDS",
    "REQUIRED_PAGE_CITATION_FIELDS",
    "SOURCES_HEADING",
    "CitationContext",
    "CitationRef",
    "CitationSource",
    "CitationWarning",
    "CitedAnswer",
    "CitedClaim",
    "build_citation_context",
    "build_citation_sources",
    "build_cited_answer",
    "build_cited_answer_from_claim_specs",
    "cite_claim",
    "citation_target_for_attachment",
    "citation_target_for_code",
    "citation_target_for_metadata",
    "citation_target_for_page",
    "format_answer_with_sources",
    "format_cited_answer_markdown",
    "format_claim_markdown",
    "format_sources_markdown",
    "missing_citation_fields",
    "missing_fields_for_document",
    "missing_page_citation_fields",
]
