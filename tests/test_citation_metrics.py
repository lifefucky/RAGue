from __future__ import annotations

from langchain_core.documents import Document

from rague.citations import (
    CitationRef,
    CitedAnswer,
    CitedClaim,
    build_citation_context,
    build_cited_answer_from_claim_specs,
    cite_claim,
)
from rague.evaluation.metrics import calculate_citation_rate


def _page_doc(*, chunk_id: str) -> Document:
    return Document(
        page_content="sample",
        metadata={
            "source_type": "confluence",
            "document_type": "page",
            "document_id": "confluence:page:131304166",
            "chunk_id": chunk_id,
            "page_id": "131304166",
            "title": "Debezium setup",
            "path": "Data/Debezium",
            "source": "https://wiki.example/pages/viewpage.action?pageId=131304166",
            "source_updated_at": "2026-06-06T10:00:00+00:00",
            "ingested_at": "2026-06-06T11:00:00+00:00",
        },
        id=chunk_id,
    )


def test_calculate_citation_rate_all_claims_cited() -> None:
    context = build_citation_context(
        [
            _page_doc(chunk_id="chunk-1"),
            _page_doc(chunk_id="chunk-2"),
        ]
    )
    answer = build_cited_answer_from_claim_specs(
        [
            ("Claim one.", ["chunk-1"]),
            ("Claim two.", ["chunk-2"]),
        ],
        context,
    )

    assert calculate_citation_rate(answer) == 1.0


def test_calculate_citation_rate_partial_citations() -> None:
    context = build_citation_context([_page_doc(chunk_id="chunk-1")])
    cited_claim, _ = cite_claim("Cited claim.", ["chunk-1"], context)
    answer = CitedAnswer(
        claims=[
            cited_claim,
            CitedClaim(text="Uncited claim."),
        ],
        sources=context.sources,
    )

    assert calculate_citation_rate(answer) == 0.5


def test_calculate_citation_rate_empty_claims_returns_zero() -> None:
    answer = CitedAnswer(claims=[])

    assert calculate_citation_rate(answer) == 0.0


def test_calculate_citation_rate_ignores_unknown_chunk_warnings() -> None:
    context = build_citation_context([_page_doc(chunk_id="chunk-1")])
    claim, _ = cite_claim("Partial.", ["chunk-1", "missing"], context)
    answer = CitedAnswer(claims=[claim], sources=context.sources)

    assert calculate_citation_rate(answer) == 1.0
    assert len(claim.citation_refs) == 1
    assert claim.citation_refs[0] == CitationRef(
        source_id="confluence:page:131304166",
        chunk_id="chunk-1",
        label="[1]",
        document_id="confluence:page:131304166",
        source="https://wiki.example/pages/viewpage.action?pageId=131304166",
        metadata=claim.citation_refs[0].metadata,
    )
