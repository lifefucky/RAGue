from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from rague.agents.parsers import (
    DocumentRelevanceOutput,
    GeneratedAnswerOutput,
    ShouldRetrieveOutput,
)
from rague.agents.workflows import (
    AgentWorkflowConfig,
    GeneratedAnswer,
    RelevanceDecision,
)
from rague.evaluation.agent_trace import (
    build_mock_traced_runner,
    run_traced_agent_case_with_bundle,
    run_traced_agent_evaluation_cases,
)
from rague.evaluation.dataset import EvaluationCase
from rague.evaluation.tracing import (
    TraceRecorder,
    default_trace_output_path,
    render_trace_summary_markdown,
    safe_run_metadata,
    sanitize_task_variables,
    summarize_document_row,
    summarize_retrieval_attempt,
    read_trace_jsonl,
    write_trace_jsonl,
)


def _page_doc(
    *,
    chunk_id: str,
    page_id: str,
    title: str,
    text: str = "sample",
    rerank_score: float | None = None,
    vector_score: float | None = None,
    bm25_score: float | None = None,
    retrieval_sources: list[str] | None = None,
) -> Document:
    metadata: dict[str, object] = {
        "chunk_id": chunk_id,
        "page_id": page_id,
        "document_id": f"confluence:page:{page_id}",
        "title": title,
    }
    if rerank_score is not None:
        metadata["rerank_score"] = rerank_score
    if vector_score is not None:
        metadata["vector_score"] = vector_score
    if bm25_score is not None:
        metadata["bm25_score"] = bm25_score
    if retrieval_sources is not None:
        metadata["retrieval_sources"] = retrieval_sources
    return Document(page_content=text, metadata=metadata, id=chunk_id)


def _sample_case() -> EvaluationCase:
    return EvaluationCase(
        id="dq-uuid-sql",
        question="Какой SQL-запрос используется для проверки guid?",
        expected_answer_contains=["guid", "uuid"],
        relevant_docs=["131304575"],
        should_retrieve=True,
        should_cite=True,
        relevant_id_field="page_id",
    )


def test_summarize_document_row_marks_relevant_pages() -> None:
    document = _page_doc(
        chunk_id="confluence:page:131304575:v8:code:1",
        page_id="131304575",
        title="Asmodeus DQ",
        rerank_score=0.91,
        vector_score=0.82,
        bm25_score=4.2,
        retrieval_sources=["vector", "bm25"],
    )

    row = summarize_document_row(
        document,
        rank=1,
        id_field="page_id",
        relevant_ids={"131304575"},
    )

    assert row["eval_id"] == "131304575"
    assert row["is_relevant"] is True
    assert row["title"] == "Asmodeus DQ"
    assert row["rerank_score"] == 0.91
    assert row["vector_score"] == 0.82
    assert row["bm25_score"] == 4.2
    assert row["retrieval_sources"] == ["vector", "bm25"]


def test_summarize_document_row_omits_missing_scores() -> None:
    document = _page_doc(
        chunk_id="confluence:page:131304575:v8:code:1",
        page_id="131304575",
        title="Asmodeus DQ",
    )

    row = summarize_document_row(
        document,
        rank=1,
        id_field="page_id",
        relevant_ids={"131304575"},
    )

    assert "rerank_score" not in row
    assert "vector_score" not in row
    assert "bm25_score" not in row
    assert "retrieval_sources" not in row


def test_sanitize_task_variables_truncates_documents_context() -> None:
    sanitized = sanitize_task_variables(
        {
            "question": "test",
            "documents_context": "x" * 500,
        }
    )

    assert sanitized["question"] == "test"
    assert sanitized["documents_context"]["chars"] == 500
    assert len(sanitized["documents_context"]["preview"]) <= 200


def test_summarize_retrieval_attempt_includes_funnel_metrics() -> None:
    case = _sample_case()
    documents = [
        _page_doc(
            chunk_id="c1",
            page_id="131304174",
            title="Other",
            rerank_score=0.55,
            vector_score=0.71,
            retrieval_sources=["vector"],
        ),
        _page_doc(
            chunk_id="c2",
            page_id="131304575",
            title="Target",
            rerank_score=0.91,
            vector_score=0.82,
            bm25_score=4.2,
            retrieval_sources=["vector", "bm25"],
        ),
    ]

    attempt = summarize_retrieval_attempt(
        query=case.question,
        documents=documents,
        case=case,
        attempt_index=0,
        k_values=(1, 5),
    )

    assert attempt["retrieved_ids"] == ["131304174", "131304575"]
    assert attempt["relevant_found_in_results"] == ["131304575"]
    assert attempt["metrics"]["recall_at_5"] == 1.0
    assert attempt["documents"][0]["rerank_score"] == 0.55
    assert attempt["documents"][0]["vector_score"] == 0.71
    assert "bm25_score" not in attempt["documents"][0]
    assert attempt["documents"][0]["retrieval_sources"] == ["vector"]
    assert attempt["documents"][1]["rerank_score"] == 0.91
    assert attempt["documents"][1]["bm25_score"] == 4.2


def test_trace_recorder_records_llm_task_outputs() -> None:
    case = _sample_case()
    recorder = TraceRecorder(case=case)

    recorder.on_task_output(
        "should_retrieve",
        {"question": case.question},
        ShouldRetrieveOutput(needs_retrieval=True, reason="Needs SQL from docs"),
    )
    recorder.on_task_output(
        "grade_documents",
        {"query": case.question, "documents_context": "docs"},
        DocumentRelevanceOutput(is_relevant=True, reason="Relevant SQL found"),
    )
    recorder.on_task_output(
        "generate_answer",
        {"question": case.question},
        GeneratedAnswerOutput(
            answer_text="guid and uuid query",
            claims=[],
        ),
    )

    types = [step["type"] for step in recorder.steps]
    assert types == ["routing", "grade_documents", "generate_answer"]
    assert recorder.steps[0]["reason"] == "Needs SQL from docs"
    assert recorder.steps[1]["is_relevant"] is True


def test_safe_run_metadata_excludes_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv("QDRANT_COLLECTION", "confluence_pages_e5_v1")

    metadata = safe_run_metadata()

    assert "OPENAI_API_KEY" not in metadata
    assert metadata["qdrant_collection"] == "confluence_pages_e5_v1"


def test_mock_traced_runner_records_retrieval_and_final_metrics() -> None:
    case = _sample_case()
    recorder = TraceRecorder(case=case)

    def should_retrieve(question: str) -> bool:
        recorder.on_task_output(
            "should_retrieve",
            {"question": question},
            ShouldRetrieveOutput(needs_retrieval=True, reason="lookup"),
        )
        return True

    def grade_documents(query: str, documents: list[Document]) -> RelevanceDecision:
        recorder.on_task_output(
            "grade_documents",
            {"query": query, "documents_context": "docs"},
            DocumentRelevanceOutput(is_relevant=True, reason="ok"),
        )
        return RelevanceDecision(is_relevant=True, reason="ok")

    run_agent = build_mock_traced_runner(
        recorder=recorder,
        retriever=lambda query: [
            _page_doc(chunk_id="c2", page_id="131304575", title="Target", text=query)
        ],
        should_retrieve=should_retrieve,
        grade_documents=grade_documents,
        generate_answer=lambda question, docs, context: GeneratedAnswer(
            answer_text="answer with guid and uuid"
        ),
        rewrite_query=lambda question, query, docs: query,
        config=AgentWorkflowConfig(max_rewrites=0),
    )

    trace = run_traced_agent_case_with_bundle(case, run_agent, recorder=recorder)

    step_types = [step["type"] for step in trace["steps"]]
    assert "retrieval" in step_types
    assert "routing" in step_types
    assert "grade_documents" in step_types
    assert "workflow_finished" in step_types
    assert trace["retrieval_attempts"][0]["relevant_found_in_results"] == ["131304575"]
    metrics_step = next(step for step in trace["steps"] if step["type"] == "metrics")
    assert metrics_step["contains_score"] == 1.0


def test_run_traced_agent_evaluation_cases_aggregates_metrics() -> None:
    case = _sample_case()

    def run_case(eval_case: EvaluationCase) -> dict:
        recorder = TraceRecorder(case=eval_case)
        recorder.record_retrieval(
            eval_case.question,
            [_page_doc(chunk_id="c2", page_id="131304575", title="Target")],
        )
        recorder.on_task_output(
            "should_retrieve",
            {"question": eval_case.question},
            ShouldRetrieveOutput(needs_retrieval=True, reason="lookup"),
        )
        recorder.record_final_state(
            {
                "should_retrieve": True,
                "query": eval_case.question,
                "retry_count": 0,
                "answer": "answer with guid and uuid",
                "cited_answer": None,
            }
        )
        recorder.record_metrics(
            {
                "contains_score": 1.0,
                "citation_rate": 0.0,
                "citation_compliant": False,
                "routing_correct": True,
            }
        )
        trace = recorder.to_dict()
        trace["final_state"] = {
            "should_retrieve": True,
            "answer": "answer with guid and uuid",
            "cited_answer": None,
        }
        return trace

    results = run_traced_agent_evaluation_cases([case], run_case=run_case)

    assert results["case_count"] == 1
    assert results["routing"]["accuracy"] == 1.0
    assert results["generation"]["answer_contains_accuracy"] == 1.0
    assert "timestamp_utc" in results["run_metadata"]


def test_write_trace_jsonl_and_default_path(tmp_path: Path) -> None:
    traces = [{"case": {"case_id": "demo"}, "steps": []}]
    output = tmp_path / "trace.jsonl"

    write_trace_jsonl(output, traces)

    content = output.read_text(encoding="utf-8")
    assert '"case_id": "demo"' in content
    loaded = read_trace_jsonl(output)
    assert len(loaded) == 1
    assert loaded[0]["case"]["case_id"] == "demo"

    generated = default_trace_output_path(runs_dir=tmp_path / "runs")
    assert generated.parent.name == "runs"
    assert generated.name.endswith("_agent_trace.jsonl")


def test_render_trace_summary_markdown_includes_case_diagnostics() -> None:
    trace = {
        "case": {
            "case_id": "dq-uuid-sql",
            "question": "SQL guid?",
        },
        "retrieval_attempts": [
            {
                "relevant_found_in_results": ["131304575"],
                "metrics": {"recall_at_10": 1.0},
            }
        ],
        "steps": [
            {
                "type": "routing",
                "needs_retrieval": True,
                "reason": "Needs docs",
            },
            {
                "type": "grade_documents",
                "is_relevant": True,
                "reason": "Found SQL",
            },
            {
                "type": "metrics",
                "contains_score": 0.0,
                "citation_compliant": True,
            },
        ],
    }

    markdown = render_trace_summary_markdown(
        traces=[trace],
        run_metadata={"chat_model": "gpt-4o-mini"},
        aggregate={
            "routing": {"accuracy": 1.0, "correct": 1, "case_count": 1},
            "generation": {"answer_contains_accuracy": 0.0, "citation_compliance_rate": 1.0},
        },
    )

    assert "dq-uuid-sql" in markdown
    assert "Needs docs" in markdown
    assert "Found SQL" in markdown
    assert "Routing accuracy" in markdown


def test_agent_trace_cli_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rague.evaluation import cli

    dataset = tmp_path / "cases.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "greeting",
                    "question": "Привет",
                    "expected_answer_contains": None,
                    "relevant_docs": [],
                    "should_retrieve": False,
                    "should_cite": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    output_jsonl = tmp_path / "trace.jsonl"
    summary = tmp_path / "summary.md"

    def fake_run_traced(cases, *, run_case=None):
        return {
            "run_metadata": {"chat_model": "gpt-4o-mini"},
            "case_count": len(cases),
            "traces": [{"case": {"case_id": "greeting", "question": "Привет"}, "steps": []}],
            "routing": {"case_count": 1, "correct": 1, "accuracy": 1.0, "mismatches": []},
            "generation": {
                "case_count": 1,
                "answer_contains_accuracy": None,
                "citation_compliance_rate": 1.0,
                "average_citation_rate": None,
                "per_case": [],
            },
        }

    monkeypatch.setattr(cli, "run_traced_agent_evaluation_cases", fake_run_traced)

    exit_code = cli.main(
        [
            "agent-trace",
            "--dataset",
            str(dataset),
            "--limit",
            "1",
            "--output-jsonl",
            str(output_jsonl),
            "--summary",
            str(summary),
            "--json",
        ]
    )

    assert exit_code == 0
    assert output_jsonl.exists()
    assert summary.exists()
    payload = json.loads(output_jsonl.read_text(encoding="utf-8").strip())
    assert payload["case"]["case_id"] == "greeting"
