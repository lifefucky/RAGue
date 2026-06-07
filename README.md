# RAGue — Confluence to Qdrant Ingestion

Pipeline for loading corporate Confluence pages, splitting Markdown content into
chunks, embedding them, and upserting into Qdrant.

## Prerequisites

- Python 3.12+
- Docker (for local Qdrant)
- Confluence credentials (see Authentication below)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in CONFLUENCE_URL, CONFLUENCE_USERNAME, and one secret (password or API token)
```

## Authentication

| Deployment | `CONFLUENCE_USERNAME` | Secret |
|------------|----------------------|--------|
| Confluence Cloud | Atlassian account email | `CONFLUENCE_API_TOKEN` (API token from [id.atlassian.com](https://id.atlassian.com/manage-profile/security/api-tokens)) |
| Confluence Server / Data Center | Login name | `CONFLUENCE_PASSWORD` (account password) or a Personal Access Token via loader `token=` |

If both `CONFLUENCE_PASSWORD` and `CONFLUENCE_API_TOKEN` are set, the password takes
precedence. Regular account passwords do not work against Confluence Cloud REST API.

## Embeddings

Default local backend: `intfloat/multilingual-e5-base` via `sentence-transformers`
(768-dimensional, multilingual). Configure in `.env`:

```bash
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_VECTOR_SIZE=768
```

The first run downloads the model from Hugging Face (~1.1 GB). E5 models require a
`passage: ` prefix for indexed documents; the ingestion pipeline adds it automatically.

For smoke tests without GPU/model download, use `EMBEDDING_PROVIDER=deterministic`.

On Intel Mac (x86_64), `requirements.txt` pins `torch<2.3`, `transformers<5`, and
`numpy<2` because newer transformers require torch versions unavailable on that platform.

Use a dedicated Qdrant collection for 768-dimensional E5 vectors (default in
`.env.example`: `confluence_pages_e5_v1`). Do not reuse a collection created with
`EMBEDDING_VECTOR_SIZE=384` — Qdrant will reject upserts with mismatched dimensions.

Start Qdrant:

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

## Run From Python

Minimal example with explicit configuration:

```python
from rague.ingestion.confluence_to_qdrant import run_ingestion

report = run_ingestion(
    {
        "confluence_url": "https://your-confluence.example.com",
        "confluence_username": "your-username",
        "confluence_password": "your-password",  # Server/DC; omit for Cloud
        # "confluence_api_token": "your-api-token",  # Cloud instead of password
        "parent_page_id": "131302699",
        "qdrant_url": "http://localhost:6333",
        "collection_name": "confluence_pages_e5_v1",
        "embedding_provider": "sentence_transformers",
        "embedding_model": "intfloat/multilingual-e5-base",
        "embedding_vector_size": 768,
        "changelog_dir": "changelog",
        "attachment_sample_dir": "data/attachment_samples",
    }
)

print(
    f"loaded={report.pages_loaded}, "
    f"failed={report.pages_failed}, "
    f"upserted={report.points_upserted}"
)
```

Alternative scopes:

- `parent_page_id` — load a page tree recursively
- `space_key` — load pages from a Confluence space
- `page_ids` — explicit list, e.g. `["131302699", "131304370"]`

Incremental reload uses the latest `source_updated_at` already stored in Qdrant.
Use `"full_reload": True` to reload everything from scratch.

Qdrant uses HNSW for vector indexing. Override the default index parameters with
`QDRANT_HNSW_M`, `QDRANT_HNSW_EF_CONSTRUCT`, and
`QDRANT_HNSW_FULL_SCAN_THRESHOLD`, or the matching CLI flags. For retrieval
search quality/latency tuning, use `QDRANT_HNSW_EF_SEARCH`. The current default
`QDRANT_HNSW_FULL_SCAN_THRESHOLD=10` is temporary for small collections and
should be revisited after HNSW benchmarks on larger corpora.

Attachments are not indexed in the current MVP. If the ingestion run encounters
attachments, it saves at most one local sample per file extension to
`data/attachment_samples/` for future converter work. The `data/` directory is
ignored by git.

## Run From CLI

```bash
python -m rague.ingestion.confluence_to_qdrant \
  --confluence-url "$CONFLUENCE_URL" \
  --confluence-username "$CONFLUENCE_USERNAME" \
  --confluence-password "$CONFLUENCE_PASSWORD" \
  --parent-page-id "$CONFLUENCE_PARENT_PAGE_ID" \
  --attachment-sample-dir data/attachment_samples
```

After each run, a Markdown report is written to `changelog/`.

## Hybrid Retrieval

Run hybrid BM25 + vector retrieval with cross-encoder reranking:

```bash
python -m rague.retrieval.hybrid_reranker "Как настроить Debezium?"
```

Configure retrieval via `.env`:

- `RETRIEVAL_TOP_K` — final top-k after reranking
- `RETRIEVAL_BM25_LIMIT` — BM25 candidate pool size before merge/dedup
- `RETRIEVAL_VECTOR_LIMIT` — vector candidate pool size before merge/dedup
- `QDRANT_HNSW_EF_SEARCH`
- `RERANKER_MODEL` — cross-encoder model name or preset alias

Before retrieval, `create_retriever_from_config()` checks that Qdrant is
reachable, the target collection exists, and it is not empty. Retrieval does
not create collections automatically; run ingestion first. The hybrid CLI
fails fast with a clear error if Qdrant is unavailable or the collection is
missing/empty.

Filter retrieval scope via CLI flags:

```bash
python -m rague.retrieval.hybrid_reranker "Debezium setup" \
  --source-type confluence \
  --document-type page \
  --space MYSPACE \
  --page-id 131304166 \
  --current-only
```

Use `--no-current-only` to include non-current chunks.

### Cross-encoder reranker

Reranking uses `CrossEncoderReranker` from `rague/retrieval/cross_encoder_reranker.py`.
Default model: `BAAI/bge-reranker-v2-m3` (multilingual, aligned with E5).

| Preset / model | Use case |
| --- | --- |
| `bge_m3` or `BAAI/bge-reranker-v2-m3` | Default for multilingual Confluence corpus |
| `ms_marco` or `cross-encoder/ms-marco-MiniLM-L-6-v2` | Faster, English-centric baseline |

Example:

```bash
RERANKER_MODEL=ms_marco python -m rague.retrieval.hybrid_reranker "Debezium setup"
```

Cross-encoder batch size is configurable via `RERANKER_BATCH_SIZE` in `.env` or
`--reranker-batch-size` on the hybrid CLI. Use it to tune reranker latency and
memory during benchmarks. Leave the value empty to use the sentence-transformers
default (`32`).

BM25 for MVP uses in-memory `rank-bm25` over chunk text loaded from Qdrant via
`Bm25ChunkIndex.from_store()`. The index is built lazily on first BM25 retrieval
and can be rebuilt after ingestion with `retriever.refresh_bm25_index()` or
`--refresh-bm25` in the hybrid CLI. Qdrant full-text search remains a future
option for larger corpora.

Query embeddings for `intfloat/multilingual-e5-base` use separate prefixes:
`passage:` for documents and `query:` for retrieval queries via `embed_query()`.

Vector search is implemented in `QdrantChunkStore.search_similar()`. It supports
metadata filtering via `metadata_filter={...}` or a raw Qdrant `Filter`, and
query-time HNSW tuning via `hnsw_ef`.

Run opt-in Qdrant integration tests with:

```bash
RAGUE_RUN_QDRANT_INTEGRATION=1 python3 -m pytest tests/test_qdrant_vector_search.py -k integration
```

### Citations and answer transparency

Retrieved chunks can be turned into a structured cited answer with claim-level
references and a Markdown `## Источники` section. The citation layer enforces a
page-specific metadata contract for Confluence pages while preserving additional
metadata pass-through for code chunks, attachments, and future document types.
Qdrant payload indexes include citation fields `title`, `path`, and `source`.
Each retrieved chunk keeps its own `citation_target` in `CitationRef.metadata`,
so code and attachment-like details are not lost when sources are deduplicated.

```python
from rague.citations import (
    build_citation_context,
    build_cited_answer_from_claim_specs,
    format_cited_answer_markdown,
)
from rague.evaluation.metrics import calculate_citation_rate
from rague.retrieval.hybrid_reranker import create_retriever_from_env

retriever = create_retriever_from_env()
documents = retriever.invoke("Как настроить Debezium?")
context = build_citation_context(documents)

answer = build_cited_answer_from_claim_specs(
    [
        ("Debezium connector настраивается через SQL.", [documents[0].id]),
        ("Kafka topic должен существовать до запуска.", [documents[1].id]),
    ],
    context,
)
print(format_cited_answer_markdown(answer))
print(f"citation_rate={calculate_citation_rate(answer):.2f}")
```

The citation layer provides the contract for claims + chunk ids validation and
is wired into the agent workflow generate path.

### Agentic RAG workflow

`rague/agents/workflows.py` implements a synchronous LangGraph workflow with
nodes `agent`, `retrieve`, `grade_documents`, `generate`, and `rewrite_query`.
Production LLM decisions are loaded from versioned YAML prompts in
`rague/prompts/agentic_rag/` and executed through `AgentLlmDecisions`.

Configure via `.env`:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
RAGUE_CHAT_MODEL=gpt-4o-mini
RAGUE_MAX_REWRITES=2
RAGUE_CHAT_TEMPERATURE=0
RAGUE_CHAT_TIMEOUT=60
RAGUE_CHAT_MAX_RETRIES=2
RAGUE_AGENT_STREAMING=0
```

Run production workflow from CLI:

```bash
python -m rague.agents.workflows "Что такое LangGraph?"
```

Run from Python:

```python
from rague.agents import run_agentic_rag_from_env, stream_agentic_rag_from_env

state = run_agentic_rag_from_env("Что такое LangGraph?")
print(state["answer"])

for event in stream_agentic_rag_from_env("Что такое LangGraph?"):
    print(event.event_type, event.data)
```

Prompt versioning:

- Each task has its own YAML file under `rague/prompts/agentic_rag/`.
- Change `current_version` in the YAML file to switch prompt versions.
- Supported tasks: `should_retrieve`, `grade_documents`, `rewrite_query`, `generate_answer`.

Example with fake backends for local development:

```python
from rague.agents import (
    AgentWorkflowConfig,
    GeneratedAnswer,
    RelevanceDecision,
    run_agentic_rag,
)
from rague.retrieval.hybrid_reranker import create_retriever_from_env

retriever = create_retriever_from_env()

state = run_agentic_rag(
    "Что такое LangGraph?",
    retriever=retriever.invoke,
    should_retrieve=lambda question: "LangGraph" in question,
    grade_documents=lambda query, docs: RelevanceDecision(is_relevant=bool(docs)),
    generate_answer=lambda question, docs, context: (
        GeneratedAnswer(
            claim_specs=[("LangGraph — это workflow-граф.", [docs[0].id])]
        )
        if docs and context
        else GeneratedAnswer(answer_text="Привет!")
    ),
    rewrite_query=lambda question, query, docs: f"{query} (refined)",
    config=AgentWorkflowConfig(max_rewrites=2),
)

print(state["answer"])
```

Integration tests:

```bash
python3 -m pytest tests/test_agent_workflow_integration.py -q
RAGUE_RUN_AGENT_INTEGRATION=1 OPENAI_API_KEY=... python3 -m pytest tests/test_agent_workflow_integration.py -q
RAGUE_RUN_AGENT_INTEGRATION=1 RAGUE_RUN_QDRANT_INTEGRATION=1 OPENAI_API_KEY=... python3 -m pytest tests/test_agent_workflow_integration.py -q
```

Token-level LLM streaming remains optional future work; current streaming API
emits stable workflow events such as `agent_decision`, `retrieval_finished`,
`documents_graded`, `query_rewritten`, `answer_generated`, and
`workflow_finished`.

## Logging

Ingestion logs page progress to the terminal and prints a final chunk summary:

```text
INFO  Discovered 87 page(s), scope=parent_page_id=131304166
INFO  [  1/ 87] page 131304166 'Руководство' -> 12 chunks, 12 upserted (4.2s)
...
Ingestion finished in 312.5s
Pages: discovered=87 loaded=82 skipped=3 failed=2
Chunks: total=412 | per page avg=5.0 min=1 max=24
Chunks by type: code=42, code_summary=12, table_row=8, text=350
Code fragments: total=54 (code=42, code_summary=12)
Attachments: discovered=18, samples_saved=4, skipped=14, failed=0, extensions=pdf, pptx
```

Control verbosity with `--log-level` or `LOG_LEVEL` in `.env` (`DEBUG`, `INFO`, `WARNING`,
`ERROR`). The final summary is always printed; per-page progress requires `INFO` or `DEBUG`.

The Markdown changelog includes **Chunk Summary** (with code fragment counts) and
**Attachments** sections with the same aggregate metrics.

## Project Layout

- `rague/ingestion/confluence_to_qdrant.py` — orchestration entrypoint
- `rague/sources/confluence/` — Confluence loaders
- `rague/chunking/` — Markdown-aware chunking
- `rague/embeddings/` — pluggable embedding backends
- `rague/vectorstores/` — Qdrant helpers
- `rague/retrieval/` — hybrid BM25 + vector retrieval and reranking
- `rague/citations/` — citation context, claim linking, and Markdown formatting
- `rague/agents/` — Agentic RAG workflow (LangGraph + YAML prompts + production LLM adapter)
- `rague/prompts/` — versioned YAML prompt configs
- `rague/evaluation/` — RAG quality metrics including citation rate
- `docker-compose.qdrant.yml` — local Qdrant service
