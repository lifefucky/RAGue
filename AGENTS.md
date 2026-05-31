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
- `rague/ingestion/changelog.py` - Markdown changelog writer for ingestion runs.
- `rague/chunking/` - text splitting utilities.
- `rague/chunking/markdown.py` - Markdown-aware LangChain splitter that preserves header metadata.
- `rague/embeddings/` - pluggable embedding backend abstraction and factory.
- `rague/embeddings/base.py` - embedding backend protocol.
- `rague/embeddings/factory.py` - embedding backend factory selected by config or environment.
- `rague/vectorstores/` - vector store integrations.
- `rague/vectorstores/qdrant_store.py` - Qdrant collection helpers for chunk upsert and delete.
- `rague/retrieval/` - future hybrid retrieval and reranking components.
- `rague/retrieval/hybrid_reranker.py` - placeholder for `HybridRerankerRetriever`.
- `rague/agents/` - future Agentic RAG workflow components.
- `rague/agents/workflows.py` - placeholder for agent workflow graph definitions.
- `rague/evaluation/` - future RAG evaluation utilities.
- `rague/evaluation/metrics.py` - placeholder for retrieval, generation, and citation metrics.

## `docs_design/` Navigation

- `docs_design/architecture.md` - early architecture draft for RAG, hybrid retrieval, reranking, Qdrant, agentic workflow, and citation metrics.
- `docs_design/implementation_considerations.md` - practical notes on large Confluence spaces, duplicate discovery, and attachment ingestion.
- `docs_design/ingestion_plan.md` - planning notes and architectural decisions for collecting Confluence data into Qdrant.
- `docs_design/metrics.md` - notes on retrieval, reranking, generation, and citation metrics.
- `docs_design/optional_features.md` - optional future ideas for agent execution, consensus, and quality control.

## `tests/` Navigation

- `tests/__init__.py` - test package marker.
- `tests/test_ingestion_smoke.py` - smoke tests for ingestion-adjacent behavior such as attachment sample collection.
