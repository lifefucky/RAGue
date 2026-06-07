"""RAG evaluation metrics for retrieval, generation, and citations."""

from __future__ import annotations

from rague.citations.models import CitedAnswer


def calculate_citation_rate(answer: CitedAnswer) -> float:
    """Return the share of claims that have at least one citation reference."""
    if not answer.claims:
        return 0.0

    cited_count = sum(1 for claim in answer.claims if claim.citation_refs)
    return cited_count / len(answer.claims)
