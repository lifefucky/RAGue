from __future__ import annotations

from rague.evaluation.reporting import render_evaluation_summary_markdown


def test_render_evaluation_summary_markdown_includes_sections() -> None:
    markdown = render_evaluation_summary_markdown(
        {
            "retrieval": {
                "case_count": 2,
                "mrr": 0.75,
                "precision_at_k": {1: 0.5},
                "recall_at_k": {1: 0.5},
            },
            "routing": {
                "case_count": 2,
                "accuracy": 1.0,
                "mismatches": [],
            },
            "generation": {
                "case_count": 2,
                "answer_contains_accuracy": 0.5,
                "citation_compliance_rate": 1.0,
                "average_citation_rate": 0.8,
            },
        }
    )

    assert "# Evaluation Summary" in markdown
    assert "## Retrieval" in markdown
    assert "## Routing" in markdown
    assert "## Generation" in markdown
    assert "MRR" in markdown
