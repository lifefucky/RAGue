# Project Guidance

## Loader Placement

Specialized loaders for a concrete external source must live under
`rague/sources/`. Use a separate subdirectory per source, for example
`rague/sources/confluence/`.

## Navigation Maintenance

When adding a new directory, add it to the navigation below with a short
description of its contents and purpose.

## Root Navigation

- `README.md` - setup and run instructions for Confluence to Qdrant ingestion.
- `requirements.txt` - Python dependencies for the ingestion pipeline.
- `docker-compose.qdrant.yml` - local Qdrant service for development.
- `.env.example` - safe template for local environment variables.
- `changelog/` - generated ingestion run reports; Markdown reports are ignored by git.
- `data/evaluation/` - corpus-bound labeled evaluation datasets for opt-in Qdrant/LLM runs.
- `data/evaluation/runs/` - pretty-printed JSON trace artifacts (`indent=2`, blank line between cases) from `agent-trace` evaluation runs.

## `rague/` Navigation

- `rague/__init__.py` - Python package marker.
- `rague/sources/` - source-specific loaders.
- `rague/sources/confluence/` - corporate Confluence loaders.
- `rague/sources/confluence/loader.py` - single-page Confluence loader and HTML-to-Markdown helpers.
- `rague/sources/confluence/multi_page_loader.py` - multi-page Confluence loader for parent-page, space-key, and explicit page-id scopes.
- `rague/ingestion/` - ingestion orchestration workflows.
- `rague/ingestion/confluence_to_qdrant.py` - end-to-end loader + splitter + embedder + Qdrant upsert entrypoint.
- `rague/ingestion/changelog.py` - Markdown changelog writer and chunk statistics for ingestion runs.
- `rague/ingestion/logging_config.py` - terminal logging setup for ingestion CLI progress output.
- `rague/chunking/` - text splitting utilities.
- `rague/chunking/markdown.py` - Markdown-aware LangChain splitter that preserves header metadata.
- `rague/embeddings/` - pluggable embedding backend abstraction and factory.
- `rague/embeddings/base.py` - embedding backend protocol.
- `rague/embeddings/factory.py` - embedding backend factory selected by config or environment.
- `rague/vectorstores/` - vector store integrations.
- `rague/vectorstores/qdrant_store.py` - Qdrant collection helpers for chunk upsert, delete, and vector search.
- `rague/retrieval/` - hybrid retrieval and reranking components.
- `rague/retrieval/bm25_index.py` - in-memory BM25 index over Qdrant chunk corpus.
- `rague/retrieval/cross_encoder_reranker.py` - `CrossEncoderReranker` wrapper for query-document scoring with configurable model presets.
- `rague/retrieval/hybrid_reranker.py` - `HybridRerankerRetriever` with parallel BM25/vector search and cross-encoder reranking.
- `rague/citations/` - citation utilities for retrieved chunk metadata and answer transparency.
- `rague/citations/models.py` - `CitationSource`, `CitationRef`, `CitationContext`, `CitedClaim`, `CitedAnswer`, and citation contract helpers.
- `rague/citations/adapters.py` - type-specific citation target adapters for page, code, and attachment chunks.
- `rague/citations/builders.py` - citation source extraction and `CitationContext` preparation from retrieved `Document` objects.
- `rague/citations/answers.py` - helpers for claim-level citation linking and structured cited answer assembly.
- `rague/citations/formatters.py` - Markdown formatters for cited claims, answer text, and `Источники` section.
- `rague/prompts/` - versioned YAML prompt configs for agent tasks.
- `rague/prompts/agentic_rag/` - one YAML file per agent task (`should_retrieve`, `grade_documents`, `rewrite_query`, `generate_answer`).
- `rague/agents/` - Agentic RAG workflow components.
- `rague/agents/workflows.py` - LangGraph workflow, production env entrypoints, streaming events, and CLI.
- `rague/agents/llm.py` - OpenAI-compatible chat model factory from environment variables.
- `rague/agents/prompts.py` - YAML prompt loader, chat prompt builder, and document context formatter.
- `rague/agents/parsers.py` - structured output schemas and JSON fallback parsers for agent decisions.
- `rague/agents/decisions.py` - `AgentLlmDecisions` adapter mapping YAML prompts + LLM outputs to workflow callables.
- `rague/evaluation/` - RAG evaluation utilities for retrieval, routing, generation, citations, HNSW benchmarks, and reporting.
- `rague/evaluation/metrics.py` - retrieval and generation metrics including `Precision@K`, `Recall@K`, `MRR`, `NDCG`, citation rate, answer contains score, and citation compliance.
- `rague/evaluation/dataset.py` - `EvaluationCase` model and JSON dataset loader for labeled evaluation questions.
- `rague/evaluation/runner.py` - `evaluate_retrieval_cases()` runner over a callable question-to-doc-ids interface.
- `rague/evaluation/retrieval.py` - document id adapters and `evaluate_retriever_cases()` for LangChain retrievers.
- `rague/evaluation/routing.py` - `evaluate_should_retrieve_cases()` for agent routing accuracy.
- `rague/evaluation/generation.py` - lightweight generation evaluation with answer contains and citation compliance checks.
- `rague/evaluation/agent.py` - agent end-to-end evaluation wrapper over workflow state.
- `rague/evaluation/agent_trace.py` - traced agent evaluation runner with per-case retrieval funnel and LLM rationale logging.
- `rague/evaluation/tracing.py` - trace schema, document summarization, JSONL writer, and Markdown trace summaries.
- `rague/evaluation/hnsw_benchmark.py` - opt-in HNSW recall/latency benchmark harness.
- `rague/evaluation/ragas_eval.py` - optional RAGAS wrapper for faithfulness and answer relevance.
- `rague/evaluation/reporting.py` - Markdown summary renderer for evaluation runs.
- `rague/evaluation/cli.py` - CLI entrypoint for retrieval, agent, agent-trace, and HNSW benchmark commands.
- `data/evaluation/` - corpus-bound labeled evaluation datasets for opt-in Qdrant/LLM runs.

## `docs_design/` Navigation

- `docs_design/architecture.md` - early architecture draft for RAG, hybrid retrieval, reranking, Qdrant, agentic workflow, and citation metrics.
- `docs_design/evaluations/` - evaluation iteration records: one Markdown file per experiment with config, metrics, and delta from previous iteration.
- `docs_design/evaluations/README.md` - naming convention (`NNN_description.md`), required sections, and iteration index.
- `docs_design/evaluations/001_baseline-hybrid-agentic-rag.md` - iteration 001 baseline: E5 embeddings, bge reranker, hybrid retrieval, citations, agentic workflow.
- `docs_design/evaluations/002_step-4-evaluation-baseline.md` - iteration 002: step 4 evaluation infrastructure, dataset, runners, CLI, and opt-in benchmark setup.
- `docs_design/implementation_considerations.md` - practical notes on large Confluence spaces, duplicate discovery, and attachment ingestion.
- `docs_design/ingestion_logging.md` - ingestion progress logging, terminal summary, chunk/attachment stats, and changelog reporting rules.
- `docs_design/ingestion_plan.md` - planning notes and architectural decisions for collecting Confluence data into Qdrant.
- `docs_design/metrics.md` - notes on retrieval, reranking, generation, and citation metrics.
- `docs_design/optional_features.md` - optional future ideas for agent execution, consensus, and quality control.

## `tests/` Navigation

- `tests/__init__.py` - test package marker.
- `tests/test_ingestion_smoke.py` - smoke tests for ingestion-adjacent behavior such as attachment sample collection.
- `tests/test_chunk_run_stats.py` - unit tests for ingestion chunk statistics and changelog chunk summary rendering.
- `tests/test_hybrid_retriever.py` - unit tests for hybrid retrieval merge, dedup, and rerank ordering.
- `tests/test_cross_encoder_reranker.py` - unit tests for cross-encoder scoring, presets, and env factory.
- `tests/test_qdrant_vector_search.py` - unit and opt-in integration tests for Qdrant vector search and retrieval health checks.
- `tests/test_bm25_index.py` - unit tests for BM25 tokenizer, ranking, refresh, and metadata preservation.
- `tests/test_citations.py` - unit tests for citation context, claim linking, metadata pass-through, and Markdown formatting.
- `tests/test_citation_metrics.py` - unit tests for citation rate metric on structured cited answers.
- `tests/fixtures/evaluation/` - local labeled evaluation dataset fixtures without Confluence or Qdrant dependencies.
- `tests/test_evaluation_metrics.py` - unit tests for retrieval metric functions and edge cases.
- `tests/test_evaluation_dataset.py` - unit tests for evaluation dataset loader and fixture validation.
- `tests/test_evaluation_runner.py` - unit tests for deterministic retrieval evaluation runner aggregation.
- `tests/test_evaluation_fixture_retrieval_validation.py` - opt-in Qdrant validation that fixture questions retrieve labeled pages.
- `tests/test_evaluation_retrieval.py` - unit tests for retriever document-id adapters.
- `tests/test_evaluation_retrieval_integration.py` - opt-in Qdrant retrieval evaluation tests.
- `tests/test_evaluation_routing.py` - unit tests for `should_retrieve` routing evaluation.
- `tests/test_evaluation_generation.py` - unit tests for generation correctness and citation compliance.
- `tests/test_evaluation_agent.py` - unit tests for agent end-to-end evaluation wrapper.
- `tests/test_evaluation_tracing.py` - unit tests for agent trace schema, traced runner, and agent-trace CLI output.
- `tests/test_evaluation_agent_integration.py` - opt-in live agent evaluation smoke tests.
- `tests/test_evaluation_reporting.py` - unit tests for evaluation Markdown reporting.
- `tests/test_evaluation_ragas.py` - unit tests for optional RAGAS wrapper behavior.
- `tests/test_hnsw_benchmark.py` - unit tests for HNSW benchmark harness.
- `tests/test_hnsw_benchmark_integration.py` - opt-in HNSW benchmark integration tests.
- `tests/test_agent_workflow.py` - unit and smoke tests for agent workflow routing, rewrite limits, retrieval tool wrapper, and cited answer generation.
- `tests/test_agent_prompts.py` - unit tests for YAML prompt loader and document context formatting.
- `tests/test_agent_parsers.py` - unit tests for structured output parsers and chunk-id filtering.
- `tests/test_agent_llm.py` - unit tests for OpenAI-compatible chat model factory.
- `tests/test_agent_decisions.py` - unit tests for production LLM decision adapter.
- `tests/test_agent_streaming.py` - unit tests for workflow event streaming.
- `tests/test_agent_workflow_integration.py` - opt-in live LLM/Qdrant integration tests for agent workflow.
