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
- `rague/retrieval/` — future hybrid retrieval and reranking
- `rague/agents/` — future Agentic RAG workflows
- `rague/evaluation/` — future RAG quality metrics
- `docker-compose.qdrant.yml` — local Qdrant service
