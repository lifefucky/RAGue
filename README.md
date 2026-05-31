# RAGue — Confluence to Qdrant Ingestion

Pipeline for loading corporate Confluence pages, splitting Markdown content into
chunks, embedding them, and upserting into Qdrant.

## Prerequisites

- Python 3.12+
- Docker (for local Qdrant)
- Confluence API token

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN
```

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
        "confluence_api_token": "your-api-token",
        "parent_page_id": "131302699",
        "qdrant_url": "http://localhost:6333",
        "collection_name": "confluence_pages_v1",
        "embedding_provider": "deterministic",
        "embedding_vector_size": 384,
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

Attachments are not indexed in the current MVP. If the ingestion run encounters
attachments, it saves at most one local sample per file extension to
`data/attachment_samples/` for future converter work. The `data/` directory is
ignored by git.

## Run From CLI

```bash
python -m rague.ingestion.confluence_to_qdrant \
  --confluence-url "$CONFLUENCE_URL" \
  --confluence-username "$CONFLUENCE_USERNAME" \
  --confluence-api-token "$CONFLUENCE_API_TOKEN" \
  --parent-page-id "$CONFLUENCE_PARENT_PAGE_ID" \
  --attachment-sample-dir data/attachment_samples
```

After each run, a Markdown report is written to `changelog/`.

## Project Layout

- `rague/ingestion/confluence_to_qdrant.py` — orchestration entrypoint
- `rague/sources/confluence/` — Confluence loaders
- `rague/chunking/` — Markdown-aware chunking
- `rague/embeddings/` — pluggable embedding backends
- `rague/vectorstores/` — Qdrant helpers
- `rague/retrieval/` — future hybrid retrieval and reranking
- `rague/agents/` — future Agentic RAG workflows
- `rague/evaluation/` — future RAG quality metrics
- `docker-compose.qdrant.yml` — local Qdrant service
