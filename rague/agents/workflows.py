"""Agentic RAG workflow skeleton built with LangGraph."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph

from rague.citations import (
    build_citation_context,
    build_cited_answer_from_claim_specs,
    format_cited_answer_markdown,
)
from rague.citations.models import CitationContext, CitedAnswer

DEFAULT_MAX_REWRITES = 2
DEFAULT_CHAT_MODEL = "gpt-4o-mini"

NODE_AGENT = "agent"
NODE_RETRIEVE = "retrieve"
NODE_GRADE = "grade_documents"
NODE_GENERATE = "generate"
NODE_REWRITE = "rewrite_query"

RetrieverCallable = Callable[[str], list[Document]]


@dataclass(frozen=True)
class RelevanceDecision:
    """Binary relevance signal for retrieved documents."""

    is_relevant: bool
    reason: str | None = None


@dataclass
class GeneratedAnswer:
    """LLM generation output consumed by the workflow generate node."""

    answer_text: str | None = None
    claim_specs: list[tuple[str, list[str]]] | None = None
    intro: str | None = None
    summary: str | None = None


@dataclass
class AgentWorkflowConfig:
    """Runtime configuration for the agentic RAG workflow."""

    max_rewrites: int = DEFAULT_MAX_REWRITES
    chat_model: str = DEFAULT_CHAT_MODEL
    debug: bool = False


@dataclass(frozen=True)
class AgentWorkflowEvent:
    """Workflow-level streaming event emitted during agent execution."""

    event_type: str
    data: dict[str, Any]


@dataclass(frozen=True)
class AgentWorkflowBundle:
    """Compiled workflow plus the dependencies used to build it."""

    app: Any
    retriever: RetrieverCallable
    should_retrieve: ShouldRetrieveDecider
    grade_documents: DocumentGrader
    generate_answer: AnswerGenerator
    rewrite_query: QueryRewriter
    config: AgentWorkflowConfig


class AgentRagState(TypedDict, total=False):
    """LangGraph state for the agentic RAG workflow."""

    question: str
    query: str
    messages: list[BaseMessage]
    documents: list[Document]
    citation_context: CitationContext | None
    relevance_decision: RelevanceDecision | None
    answer: str
    cited_answer: CitedAnswer | None
    retry_count: int
    max_retries: int
    should_retrieve: bool


class ShouldRetrieveDecider(Protocol):
    def __call__(self, question: str) -> bool: ...


class DocumentGrader(Protocol):
    def __call__(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> RelevanceDecision: ...


class AnswerGenerator(Protocol):
    def __call__(
        self,
        question: str,
        documents: Sequence[Document],
        citation_context: CitationContext | None,
    ) -> GeneratedAnswer: ...


class QueryRewriter(Protocol):
    def __call__(
        self,
        question: str,
        query: str,
        documents: Sequence[Document],
    ) -> str: ...


def create_retrieval_tool(
    retriever: RetrieverCallable,
    *,
    name: str = "retrieve_documents",
    description: str = "Retrieve relevant document chunks for a search query.",
) -> StructuredTool:
    """Build a thin LangChain tool wrapper over hybrid retrieval."""

    def _invoke(query: str) -> list[Document]:
        return retrieve_documents(retriever, query)

    return StructuredTool.from_function(
        func=_invoke,
        name=name,
        description=description,
    )


def retrieve_documents(retriever: RetrieverCallable, query: str) -> list[Document]:
    """Run retrieval without mutating document metadata."""
    return retriever(query)


def render_generated_answer(
    generated: GeneratedAnswer,
    citation_context: CitationContext | None,
) -> tuple[str, CitedAnswer | None]:
    """Convert LLM output into plain or cited Markdown answer text."""
    if generated.claim_specs is not None and citation_context is not None:
        cited_answer = build_cited_answer_from_claim_specs(
            generated.claim_specs,
            citation_context,
            intro=generated.intro,
            summary=generated.summary,
        )
        return format_cited_answer_markdown(cited_answer), cited_answer

    if generated.answer_text:
        return generated.answer_text.strip(), None

    return "", None


def _route_after_agent(state: AgentRagState) -> str:
    if state.get("should_retrieve"):
        return NODE_RETRIEVE
    return NODE_GENERATE


def _route_after_grade(state: AgentRagState) -> str:
    decision = state.get("relevance_decision")
    if decision and decision.is_relevant:
        return NODE_GENERATE

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_REWRITES)
    if retry_count < max_retries:
        return NODE_REWRITE
    return NODE_GENERATE


def build_agentic_rag_workflow(
    *,
    retriever: RetrieverCallable,
    should_retrieve: ShouldRetrieveDecider,
    grade_documents: DocumentGrader,
    generate_answer: AnswerGenerator,
    rewrite_query: QueryRewriter,
    config: AgentWorkflowConfig | None = None,
):
    """Build and compile the synchronous agentic RAG LangGraph app."""
    workflow_config = config or AgentWorkflowConfig()

    def agent_node(state: AgentRagState) -> dict[str, Any]:
        question = state["question"]
        query = state.get("query") or question
        needs_retrieval = should_retrieve(question)
        return {
            "query": query,
            "should_retrieve": needs_retrieval,
        }

    def retrieve_node(state: AgentRagState) -> dict[str, Any]:
        query = state.get("query") or state["question"]
        documents = retrieve_documents(retriever, query)
        citation_context = build_citation_context(documents) if documents else None
        return {
            "documents": documents,
            "citation_context": citation_context,
        }

    def grade_node(state: AgentRagState) -> dict[str, Any]:
        query = state.get("query") or state["question"]
        documents = state.get("documents") or []
        decision = grade_documents(query, documents)
        return {"relevance_decision": decision}

    def rewrite_node(state: AgentRagState) -> dict[str, Any]:
        question = state["question"]
        query = state.get("query") or question
        documents = state.get("documents") or []
        new_query = rewrite_query(question, query, documents)
        return {
            "query": new_query,
            "retry_count": state.get("retry_count", 0) + 1,
            "documents": [],
            "citation_context": None,
            "relevance_decision": None,
            "should_retrieve": True,
        }

    def generate_node(state: AgentRagState) -> dict[str, Any]:
        question = state["question"]
        documents = state.get("documents") or []
        citation_context = state.get("citation_context")
        generated = generate_answer(question, documents, citation_context)
        answer_text, cited_answer = render_generated_answer(
            generated,
            citation_context,
        )
        return {
            "answer": answer_text,
            "cited_answer": cited_answer,
        }

    graph = StateGraph(AgentRagState)
    graph.add_node(NODE_AGENT, agent_node)
    graph.add_node(NODE_RETRIEVE, retrieve_node)
    graph.add_node(NODE_GRADE, grade_node)
    graph.add_node(NODE_GENERATE, generate_node)
    graph.add_node(NODE_REWRITE, rewrite_node)

    graph.add_edge(START, NODE_AGENT)
    graph.add_conditional_edges(
        NODE_AGENT,
        _route_after_agent,
        {NODE_RETRIEVE: NODE_RETRIEVE, NODE_GENERATE: NODE_GENERATE},
    )
    graph.add_edge(NODE_RETRIEVE, NODE_GRADE)
    graph.add_conditional_edges(
        NODE_GRADE,
        _route_after_grade,
        {
            NODE_GENERATE: NODE_GENERATE,
            NODE_REWRITE: NODE_REWRITE,
        },
    )
    graph.add_edge(NODE_REWRITE, NODE_AGENT)
    graph.add_edge(NODE_GENERATE, END)

    return graph.compile()


def run_agentic_rag(
    question: str,
    *,
    retriever: RetrieverCallable,
    should_retrieve: ShouldRetrieveDecider,
    grade_documents: DocumentGrader,
    generate_answer: AnswerGenerator,
    rewrite_query: QueryRewriter,
    config: AgentWorkflowConfig | None = None,
    initial_query: str | None = None,
) -> AgentRagState:
    """Run the agentic RAG workflow synchronously and return final state."""
    workflow = build_agentic_rag_workflow(
        retriever=retriever,
        should_retrieve=should_retrieve,
        grade_documents=grade_documents,
        generate_answer=generate_answer,
        rewrite_query=rewrite_query,
        config=config,
    )
    workflow_config = config or AgentWorkflowConfig()
    initial_state = _initial_agent_state(
        question,
        config=workflow_config,
        initial_query=initial_query,
    )
    return workflow.invoke(initial_state)


def config_from_env() -> AgentWorkflowConfig:
    """Load agent workflow settings from environment variables."""
    return AgentWorkflowConfig(
        max_rewrites=int(os.getenv("RAGUE_MAX_REWRITES", str(DEFAULT_MAX_REWRITES))),
        chat_model=os.getenv("RAGUE_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        debug=os.getenv("RAGUE_AGENT_DEBUG", "").strip() == "1",
    )


def _initial_agent_state(
    question: str,
    *,
    config: AgentWorkflowConfig,
    initial_query: str | None = None,
) -> AgentRagState:
    return {
        "question": question,
        "query": initial_query or question,
        "messages": [],
        "documents": [],
        "citation_context": None,
        "relevance_decision": None,
        "answer": "",
        "cited_answer": None,
        "retry_count": 0,
        "max_retries": config.max_rewrites,
        "should_retrieve": False,
    }


def build_agentic_rag_from_env():
    """Build production agent workflow from environment-backed dependencies."""
    from rague.agents.decisions import AgentLlmDecisions
    from rague.agents.llm import create_chat_model_from_env
    from rague.retrieval.hybrid_reranker import create_retriever_from_env

    config = config_from_env()
    retriever = create_retriever_from_env()
    chat_model = create_chat_model_from_env()
    decisions = AgentLlmDecisions(chat_model)

    app = build_agentic_rag_workflow(
        retriever=retriever.invoke,
        should_retrieve=decisions.decide_should_retrieve,
        grade_documents=decisions.grade_documents,
        generate_answer=decisions.generate_answer,
        rewrite_query=decisions.rewrite_query,
        config=config,
    )
    return AgentWorkflowBundle(
        app=app,
        retriever=retriever.invoke,
        should_retrieve=decisions.decide_should_retrieve,
        grade_documents=decisions.grade_documents,
        generate_answer=decisions.generate_answer,
        rewrite_query=decisions.rewrite_query,
        config=config,
    )


def run_agentic_rag_from_env(
    question: str,
    *,
    initial_query: str | None = None,
) -> AgentRagState:
    """Run production agent workflow using env-backed retriever and LLM."""
    bundle = build_agentic_rag_from_env()
    initial_state = _initial_agent_state(
        question,
        config=bundle.config,
        initial_query=initial_query,
    )
    return bundle.app.invoke(initial_state)


def _map_stream_update_to_events(update: dict[str, Any]) -> list[AgentWorkflowEvent]:
    events: list[AgentWorkflowEvent] = []

    for node_name, node_state in update.items():
        if not isinstance(node_state, dict):
            continue

        if node_name == NODE_AGENT and "should_retrieve" in node_state:
            events.append(
                AgentWorkflowEvent(
                    event_type="agent_decision",
                    data={
                        "should_retrieve": node_state.get("should_retrieve"),
                        "query": node_state.get("query"),
                    },
                )
            )
        elif node_name == NODE_RETRIEVE:
            documents = node_state.get("documents") or []
            events.append(
                AgentWorkflowEvent(
                    event_type="retrieval_finished",
                    data={"document_count": len(documents)},
                )
            )
        elif node_name == NODE_GRADE and "relevance_decision" in node_state:
            decision = node_state.get("relevance_decision")
            events.append(
                AgentWorkflowEvent(
                    event_type="documents_graded",
                    data={
                        "is_relevant": getattr(decision, "is_relevant", None),
                        "reason": getattr(decision, "reason", None),
                    },
                )
            )
        elif node_name == NODE_REWRITE:
            events.append(
                AgentWorkflowEvent(
                    event_type="query_rewritten",
                    data={
                        "query": node_state.get("query"),
                        "retry_count": node_state.get("retry_count"),
                    },
                )
            )
        elif node_name == NODE_GENERATE and "answer" in node_state:
            events.append(
                AgentWorkflowEvent(
                    event_type="answer_generated",
                    data={
                        "answer_preview": str(node_state.get("answer", ""))[:200],
                        "has_cited_answer": node_state.get("cited_answer") is not None,
                    },
                )
            )

    return events


def stream_agentic_rag(
    question: str,
    *,
    retriever: RetrieverCallable,
    should_retrieve: ShouldRetrieveDecider,
    grade_documents: DocumentGrader,
    generate_answer: AnswerGenerator,
    rewrite_query: QueryRewriter,
    config: AgentWorkflowConfig | None = None,
    initial_query: str | None = None,
) -> Iterator[AgentWorkflowEvent]:
    """Stream workflow-level events for an agent run."""
    workflow = build_agentic_rag_workflow(
        retriever=retriever,
        should_retrieve=should_retrieve,
        grade_documents=grade_documents,
        generate_answer=generate_answer,
        rewrite_query=rewrite_query,
        config=config,
    )
    workflow_config = config or AgentWorkflowConfig()
    initial_state = _initial_agent_state(
        question,
        config=workflow_config,
        initial_query=initial_query,
    )

    final_state: AgentRagState = dict(initial_state)
    for update in workflow.stream(initial_state, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for event in _map_stream_update_to_events(update):
            yield event
        for node_state in update.values():
            if isinstance(node_state, dict):
                final_state.update(node_state)  # type: ignore[typeddict-item]

    yield AgentWorkflowEvent(
        event_type="workflow_finished",
        data={
            "answer": final_state.get("answer", ""),
            "should_retrieve": final_state.get("should_retrieve"),
            "retry_count": final_state.get("retry_count", 0),
        },
    )


def stream_agentic_rag_from_env(
    question: str,
    *,
    initial_query: str | None = None,
) -> Iterator[AgentWorkflowEvent]:
    """Stream production workflow events using env-backed dependencies."""
    bundle = build_agentic_rag_from_env()
    yield from stream_agentic_rag(
        question,
        retriever=bundle.retriever,
        should_retrieve=bundle.should_retrieve,
        grade_documents=bundle.grade_documents,
        generate_answer=bundle.generate_answer,
        rewrite_query=bundle.rewrite_query,
        config=bundle.config,
        initial_query=initial_query,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for synchronous production agent workflow."""
    parser = argparse.ArgumentParser(description="Run agentic RAG workflow.")
    parser.add_argument("question", help="User question.")
    args = parser.parse_args(argv)

    try:
        state = run_agentic_rag_from_env(args.question)
    except Exception as error:
        print(f"Agent workflow failed: {error}", file=sys.stderr)
        return 1

    print(state.get("answer", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
