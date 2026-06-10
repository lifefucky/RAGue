"""RAG evaluation utilities for retrieval, generation, and citations."""

from rague.evaluation.agent import (
    agent_state_to_generation_answer,
    run_agent_evaluation_cases,
)
from rague.evaluation.agent_trace import (
    run_traced_agent_case,
    run_traced_agent_evaluation_cases,
)
from rague.evaluation.dataset import (
    EvaluationCase,
    case_relevant_ids,
    load_evaluation_cases,
)
from rague.evaluation.generation import (
    GeneratedEvaluationAnswer,
    evaluate_generation_cases,
)
from rague.evaluation.metrics import (
    calculate_answer_contains_score,
    calculate_citation_compliance,
    calculate_citation_rate,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)
from rague.evaluation.reporting import render_evaluation_summary_markdown
from rague.evaluation.tracing import (
    TraceRecorder,
    default_trace_output_path,
    read_trace_jsonl,
    render_trace_summary_markdown,
    write_trace_jsonl,
)
from rague.evaluation.retrieval import (
    document_id_for_evaluation,
    evaluate_retriever_cases,
    retriever_to_retrieve_ids,
)
from rague.evaluation.routing import evaluate_should_retrieve_cases
from rague.evaluation.runner import evaluate_retrieval_cases

__all__ = [
    "EvaluationCase",
    "GeneratedEvaluationAnswer",
    "agent_state_to_generation_answer",
    "calculate_answer_contains_score",
    "calculate_citation_compliance",
    "calculate_citation_rate",
    "calculate_mrr",
    "calculate_ndcg_at_k",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_reciprocal_rank",
    "case_relevant_ids",
    "document_id_for_evaluation",
    "evaluate_generation_cases",
    "evaluate_retrieval_cases",
    "evaluate_retriever_cases",
    "evaluate_should_retrieve_cases",
    "load_evaluation_cases",
    "render_evaluation_summary_markdown",
    "retriever_to_retrieve_ids",
    "run_agent_evaluation_cases",
    "run_traced_agent_case",
    "run_traced_agent_evaluation_cases",
    "TraceRecorder",
    "default_trace_output_path",
    "render_trace_summary_markdown",
    "read_trace_jsonl",
    "write_trace_jsonl",
]
