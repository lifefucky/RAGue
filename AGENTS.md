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
- `rague/evaluation/` - future RAG evaluation utilities.
- `rague/evaluation/metrics.py` - RAG evaluation metrics including citation rate for structured cited answers.

## `docs_design/` Navigation

- `docs_design/architecture.md` - early architecture draft for RAG, hybrid retrieval, reranking, Qdrant, agentic workflow, and citation metrics.
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
- `tests/test_agent_workflow.py` - unit and smoke tests for agent workflow routing, rewrite limits, retrieval tool wrapper, and cited answer generation.
- `tests/test_agent_prompts.py` - unit tests for YAML prompt loader and document context formatting.
- `tests/test_agent_parsers.py` - unit tests for structured output parsers and chunk-id filtering.
- `tests/test_agent_llm.py` - unit tests for OpenAI-compatible chat model factory.
- `tests/test_agent_decisions.py` - unit tests for production LLM decision adapter.
- `tests/test_agent_streaming.py` - unit tests for workflow event streaming.
- `tests/test_agent_workflow_integration.py` - opt-in live LLM/Qdrant integration tests for agent workflow.
