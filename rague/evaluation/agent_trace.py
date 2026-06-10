"""Traced agent evaluation runner for per-case diagnostics."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.documents import Document

from rague.agents.workflows import (
    AgentWorkflowBundle,
    AgentWorkflowConfig,
    build_agentic_rag_workflow,
    config_from_env,
)
from rague.evaluation.agent import agent_state_to_generation_answer
from rague.evaluation.dataset import EvaluationCase
from rague.evaluation.generation import evaluate_generation_cases
from rague.evaluation.metrics import (
    calculate_answer_contains_score,
    calculate_citation_compliance,
    calculate_citation_rate,
)
from rague.evaluation.routing import evaluate_should_retrieve_cases
from rague.evaluation.tracing import TraceRecorder, safe_run_metadata


def build_traced_agent_bundle_from_env(
    *,
    recorder: TraceRecorder,
) -> AgentWorkflowBundle:
    """Build production agent workflow with tracing observer attached."""
    from rague.agents.decisions import AgentLlmDecisions
    from rague.agents.llm import create_chat_model_from_env
    from rague.retrieval.hybrid_reranker import create_retriever_from_env

    config = config_from_env()
    retriever = create_retriever_from_env()
    chat_model = create_chat_model_from_env()
    decisions = AgentLlmDecisions(chat_model, observer=recorder)

    base_retriever = retriever.invoke

    def traced_retriever(query: str) -> list[Document]:
        documents = base_retriever(query)
        recorder.record_retrieval(query, documents)
        return documents

    app = build_agentic_rag_workflow(
        retriever=traced_retriever,
        should_retrieve=decisions.decide_should_retrieve,
        grade_documents=decisions.grade_documents,
        generate_answer=decisions.generate_answer,
        rewrite_query=decisions.rewrite_query,
        config=config,
    )
    return AgentWorkflowBundle(
        app=app,
        retriever=traced_retriever,
        should_retrieve=decisions.decide_should_retrieve,
        grade_documents=decisions.grade_documents,
        generate_answer=decisions.generate_answer,
        rewrite_query=decisions.rewrite_query,
        config=config,
    )


def run_traced_agent_case(case: EvaluationCase) -> dict[str, Any]:
    """Run one agent case and return a full trace record."""
    recorder = TraceRecorder(case=case)
    bundle = build_traced_agent_bundle_from_env(recorder=recorder)

    from rague.agents.workflows import _initial_agent_state

    initial_state = _initial_agent_state(case.question, config=bundle.config)
    started = time.perf_counter()
    final_state = bundle.app.invoke(initial_state)
    recorder.duration_ms = round((time.perf_counter() - started) * 1000, 2)
    recorder.record_final_state(final_state)

    generated = agent_state_to_generation_answer(final_state)
    contains_score = calculate_answer_contains_score(
        generated.answer_text,
        case.expected_answer_contains,
    )
    citation_compliant = calculate_citation_compliance(
        generated.cited_answer,
        should_cite=case.should_cite,
    )
    citation_rate = (
        calculate_citation_rate(generated.cited_answer)
        if generated.cited_answer is not None
        else 0.0
    )
    recorder.record_metrics(
        {
            "contains_score": contains_score,
            "citation_rate": citation_rate,
            "citation_compliant": citation_compliant,
            "routing_correct": bool(final_state.get("should_retrieve"))
            == case.should_retrieve,
        }
    )

    trace = recorder.to_dict()
    trace["final_state"] = {
        "should_retrieve": final_state.get("should_retrieve"),
        "query": final_state.get("query"),
        "retry_count": final_state.get("retry_count", 0),
        "answer": generated.answer_text,
        "cited_answer": final_state.get("cited_answer"),
    }
    return trace


def run_traced_agent_case_with_bundle(
    case: EvaluationCase,
    run_agent: Callable[[str], dict[str, Any]],
    *,
    recorder: TraceRecorder | None = None,
) -> dict[str, Any]:
    """Run a traced case using a custom agent callable (for tests)."""
    active_recorder = recorder or TraceRecorder(case=case)
    started = time.perf_counter()
    final_state = run_agent(case.question)
    active_recorder.duration_ms = round((time.perf_counter() - started) * 1000, 2)
    active_recorder.record_final_state(final_state)

    generated = agent_state_to_generation_answer(final_state)
    contains_score = calculate_answer_contains_score(
        generated.answer_text,
        case.expected_answer_contains,
    )
    citation_compliant = calculate_citation_compliance(
        generated.cited_answer,
        should_cite=case.should_cite,
    )
    citation_rate = (
        calculate_citation_rate(generated.cited_answer)
        if generated.cited_answer is not None
        else 0.0
    )
    active_recorder.record_metrics(
        {
            "contains_score": contains_score,
            "citation_rate": citation_rate,
            "citation_compliant": citation_compliant,
            "routing_correct": bool(final_state.get("should_retrieve"))
            == case.should_retrieve,
        }
    )
    return active_recorder.to_dict()


def run_traced_agent_evaluation_cases(
    cases: Sequence[EvaluationCase],
    *,
    run_case: Callable[[EvaluationCase], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run traced evaluation for multiple cases and aggregate metrics."""
    traces: list[dict[str, Any]] = []
    states_by_question: dict[str, dict[str, Any]] = {}

    case_runner = run_case or run_traced_agent_case
    for case in cases:
        trace = case_runner(case)
        traces.append(trace)
        states_by_question[case.question] = trace.get("final_state", {})

    routing_results = evaluate_should_retrieve_cases(
        cases,
        lambda question: bool(states_by_question[question].get("should_retrieve")),
    )
    generation_results = evaluate_generation_cases(
        cases,
        lambda question: agent_state_to_generation_answer(states_by_question[question]),
    )

    return {
        "run_metadata": safe_run_metadata(),
        "case_count": len(cases),
        "traces": traces,
        "routing": routing_results,
        "generation": generation_results,
    }


def build_mock_traced_runner(
    *,
    recorder: TraceRecorder,
    retriever: Callable[[str], list[Document]],
    should_retrieve: Callable[[str], bool],
    grade_documents: Callable[[str, Sequence[Document]], Any],
    generate_answer: Callable[[str, Sequence[Document], Any], Any],
    rewrite_query: Callable[[str, str, Sequence[Document]], str],
    config: AgentWorkflowConfig | None = None,
) -> Callable[[str], dict[str, Any]]:
    """Build a traced workflow runner for unit tests with explicit dependencies."""
    from rague.agents.workflows import build_agentic_rag_workflow

    def traced_retriever(query: str) -> list[Document]:
        documents = retriever(query)
        recorder.record_retrieval(query, documents)
        return documents

    workflow = build_agentic_rag_workflow(
        retriever=traced_retriever,
        should_retrieve=should_retrieve,
        grade_documents=grade_documents,
        generate_answer=generate_answer,
        rewrite_query=rewrite_query,
        config=config,
    )

    from rague.agents.workflows import _initial_agent_state

    workflow_config = config or AgentWorkflowConfig()

    def run_question(question: str) -> dict[str, Any]:
        initial_state = _initial_agent_state(question, config=workflow_config)
        return workflow.invoke(initial_state)

    return run_question
