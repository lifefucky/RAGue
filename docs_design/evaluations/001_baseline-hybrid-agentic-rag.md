# Iteration 001 — Baseline Hybrid Agentic RAG

## Summary

Первая зафиксированная конфигурация end-to-end RAGue: Confluence ingestion,
hybrid retrieval (BM25 + vector + cross-encoder reranking), citation layer и
agentic LangGraph workflow. Первый labeled evaluation проведён на `data/evaluation/basic_cases.json`
(14 retrieval cases + 1 greeting). Retrieval baseline измерен; agent smoke (2 cases),
agent CLI (`--limit 3`) и полный agent-trace (15 cases) прогнаны с `gpt-4o-mini`
через `api.proxyapi.ru`. Routing и citation compliance — 100%; answer contains —
**100%** на полном agent-trace (14/14 retrieval cases).

## Changes From Previous Iteration

Baseline. Предыдущей итерации нет.

## Configuration

### Architecture Patterns

| Pattern | Implementation |
| --- | --- |
| Layered pipeline | `sources` → `chunking` → `embeddings` → `vectorstores` → `retrieval` → `citations` → `agents` |
| Source adapter | Confluence loaders в `rague/sources/confluence/`; scopes: `parent_page_id`, `space_key`, `page_ids` |
| Factory backends | `create_embedder()`, `create_chat_model_from_env()`, `create_retriever_from_env()` |
| Qdrant store wrapper | `QdrantChunkStore`: collection, HNSW, payload indexes, stable point IDs, health check |
| Metadata contract | `chunk_id`, `document_id`, `page_id`, citation fields — общий boundary между ingestion/retrieval/citations |
| Hybrid retrieval | parallel BM25 + vector → dedup → cross-encoder rerank → top-k |
| Lazy BM25 index | `rank-bm25` in-memory over Qdrant scroll; simple regex tokenizer |
| Citation contract | claim-level linking, `## Источники`, integrated in agent generate path |
| LangGraph workflow | `agent` → `retrieve` → `grade_documents` → `generate` / `rewrite_query` loop |
| Versioned prompts | YAML в `rague/prompts/agentic_rag/` + `AgentLlmDecisions` |
| Opt-in integration tests | `RAGUE_RUN_QDRANT_INTEGRATION=1`, `RAGUE_RUN_AGENT_INTEGRATION=1` |

### Models

| Role | Provider / Model | Notes |
| --- | --- | --- |
| Embeddings | `sentence_transformers` / `intfloat/multilingual-e5-base` | 768-dim, Cosine; E5 prefixes `passage:` / `query:` |
| Reranker | `BAAI/bge-reranker-v2-m3` (preset `bge_m3`) | default cross-encoder |
| Reranker (eval run) | `cross-encoder/ms-marco-MiniLM-L-6-v2` (preset `ms_marco`) | used for first retrieval baseline |
| Agent LLM | OpenAI-compatible / `gpt-4o-mini` via `api.proxyapi.ru` | orchestration only |
| Smoke embedder | `deterministic` hash embedder | tests without model download |

### Key Parameters

| Parameter | Value |
| --- | --- |
| `QDRANT_COLLECTION` | `confluence_pages_e5_v1` |
| `RETRIEVAL_TOP_K` | 10 |
| `RETRIEVAL_BM25_LIMIT` | 50 |
| `RETRIEVAL_VECTOR_LIMIT` | 50 |
| `QDRANT_HNSW_FULL_SCAN_THRESHOLD` | 10 (temporary for small corpus) |
| `QDRANT_HNSW_EF_SEARCH` | 128 |
| `RAGUE_MAX_REWRITES` | 2 |
| `chunk_size` / `chunk_overlap` | 1200 / 150 |

### Out Of Scope

- attachment indexing (only local samples in `data/attachment_samples/`);
- Qdrant full-text search (future option);
- token-level LLM streaming.

## Evaluation Setup

| Item | Value |
| --- | --- |
| Evaluation date | 2026-06-09 (retrieval baseline: 2026-06-08) |
| Labeled dataset | `data/evaluation/basic_cases.json` — 15 cases (14 retrieval, 1 greeting) |
| Corpus scope | 8 Confluence pages (51 chunks), `parent_page_id=131304166` → `confluence_pages_e5_v1` |
| Retrieval eval command | `python -m rague.evaluation retrieval --dataset data/evaluation/basic_cases.json --json` |
| Agent eval command | `set -a && source .env && set +a && python -m rague.evaluation agent --dataset data/evaluation/basic_cases.json --limit 3 --json` |
| Agent trace command (full) | `set -a && source .env && set +a && python -m rague.evaluation agent-trace --dataset data/evaluation/basic_cases.json --output-jsonl data/evaluation/runs/scores_full_agent_trace.jsonl --summary data/evaluation/runs/scores_full_agent_trace_summary.md` |
| Latest trace artifact | `data/evaluation/runs/scores_full_agent_trace.jsonl` (2026-06-09T20:40:46Z) |
| Required env (retrieval) | `QDRANT_COLLECTION=confluence_pages_e5_v1`, `EMBEDDING_PROVIDER=sentence_transformers`, `EMBEDDING_MODEL=intfloat/multilingual-e5-base`, `EMBEDDING_VECTOR_SIZE=768`, `RERANKER_MODEL=ms_marco` |
| Required env (agent) | `OPENAI_API_KEY`, `OPENAI_BASE_URL` (`.env` not auto-loaded — export before run) |
| Unit tests | `python -m pytest tests/` (component coverage) |
| Qdrant integration | `RAGUE_RUN_QDRANT_INTEGRATION=1 python -m pytest tests/test_evaluation_retrieval_integration.py` |
| Agent integration smoke | `RAGUE_RUN_AGENT_INTEGRATION=1 RAGUE_RUN_QDRANT_INTEGRATION=1 python -m pytest tests/test_evaluation_agent_integration.py` |

## Evaluation Results

### Retrieval Metrics

Hybrid retriever (`ms_marco` reranker), 14 retrieval cases, `page_id` as relevant id.

| Metric | @1 | @3 | @5 | @10 | Notes |
| --- | --- | --- | --- | --- | --- |
| Precision@K | 0.50 | 0.38 | 0.36 | 0.28 | |
| Recall@K | 0.43 | 0.71 | 0.89 | **1.00** | all relevant pages in top-10 |
| MRR | — | — | — | — | **0.66** |
| NDCG | 0.50 | 0.78 | 1.01 | 1.30 | |

Weak cases (relevant page not in top-5): `pipe-ddl-pg-exttable` (RR=0.14, recall@5=0);
`multi-dq-dds` (multi-page, RR=0.33, recall@5=0.5).

### Reranking Metrics

| Metric | Value | Notes |
| --- | --- | --- |
| MRR (post-rerank) | — | not measured |
| NDCG (post-rerank) | — | not measured |
| `bge_m3` vs `ms_marco` delta | — | not compared |

### Generation Metrics

Agent trace full dataset (15 cases; 14 with `expected_answer_contains`; artifact:
`scores_full_agent_trace.jsonl`, 2026-06-09).

| Metric | Value | Target (see `metrics.md`) | Notes |
| --- | --- | --- | --- |
| Faithfulness | — | ≥ 0.85 | not measured (RAGAS opt-in) |
| Answer Relevance | — | ≥ 0.90 | not measured |
| Answer contains accuracy | **1.00** | — | 14/14 retrieval cases matched `expected_answer_contains` |
| Citation compliance rate | **1.00** | — | 15/15 cases structurally valid |
| Average citation rate | **1.00** | ≥ 0.60 | 14/14 cited cases; all claims linked when `should_cite=true` |

### Routing Metrics

| Run | Cases | Routing accuracy | Mismatches |
| --- | --- | --- | --- |
| Smoke (`greeting`, `dq-uuid-sql`) | 2 | **1.00** | none |
| CLI `--limit 3` | 3 | **1.00** | none |
| Agent trace full dataset | 15 | **1.00** | none |

### Agent / Smoke

| Scenario | Result | Notes |
| --- | --- | --- |
| Simple question, no retrieval | pass (unit) | `tests/test_agent_workflow.py` |
| Question with retrieval | pass (unit, mocked) | routing and tool wrapper |
| Rewrite loop limit | pass (unit) | `RAGUE_MAX_REWRITES=2` |
| `langchain_openai` + `langgraph` | pass | installed in `.venv` |
| Agent integration smoke (`greeting` + `dq-uuid-sql`) | **pass** | ~62 s; routing 2/2; citation compliance 1.0 |
| Agent CLI `--limit 3` | **pass** | ~93 s; routing 3/3; citation compliance 1.0 |
| Agent trace full dataset (15 cases) | **pass** | ~284 s; routing 15/15; answer contains 14/14; citation compliance 1.0 |

### Qualitative Notes

- Hybrid retrieval, citations и agent workflow реализованы и покрыты unit-тестами.
- Retrieval recall@10 = 1.0 на всех 14 cases, но precision@1 = 0.5 — reranker часто
  ставит нерелевантные страницы выше целевой.
- Agent routing корректен (100%), citations оформлены по контракту; на полном
  agent-trace все 14 retrieval cases проходят `expected_answer_contains`.
- Code lookup cases: в trace видно разворачивание `raw_code` в контекст и ответ;
  ранний прогон `--limit 3` давал answer contains = 0% до доработок generate prompt
  и code expansion.
- BM25 tokenizer без русской морфологии — потенциальный риск для lexical recall.
- `.env` не подгружается автоматически — перед eval нужен `source .env` или export.

## Conclusion

**Quality acceptable:** partial.

Retrieval: recall@10 = 1.0, MRR = 0.66, precision@1 = 0.5 — основной риск
остаётся в ranking. Agent: routing, citations и answer contains = 100% на полном
dataset (14/14); RAGAS faithfulness/relevance не измерялись.
Итерация служит точкой отсчёта для reranker comparison и дальнейшего HNSW/RAGAS eval.

## Next Step

1. Сравнить `bge_m3` vs `ms_marco` на том же dataset → iteration `002`.
2. Улучшить precision@1 / MRR: reranker threshold, BM25 tokenizer с русской морфологией.
3. Прогнать RAGAS faithfulness / answer relevance на agent answers.
4. HNSW benchmark на текущем корпусе и пересмотреть `full_scan_threshold`.
