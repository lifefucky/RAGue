"""End-to-end Confluence ingestion pipeline into Qdrant."""

from __future__ import annotations

import argparse
import os
import traceback
from datetime import datetime, timezone
from typing import Any

from rague.chunking import MarkdownDocumentTextSplitter
from rague.embeddings.factory import create_embedder
from rague.ingestion.changelog import IngestionRunReport
from rague.sources.confluence.multi_page_loader import (
    DEFAULT_UPDATED_AFTER,
    ConfluenceMultiPageLoader,
)
from rague.vectorstores.qdrant_store import (
    DEFAULT_COLLECTION,
    QdrantChunkStore,
    enrich_chunk_metadata,
)


def run_ingestion(config: dict[str, Any]) -> IngestionRunReport:
    scope = _describe_scope(config)
    embedder = create_embedder(
        provider=config.get("embedding_provider"),
        model_name=config.get("embedding_model"),
        vector_size=config.get("embedding_vector_size"),
    )

    report = IngestionRunReport(
        scope=scope,
        collection_name=config.get("collection_name", DEFAULT_COLLECTION),
        embedding_provider=config.get("embedding_provider")
        or os.getenv("EMBEDDING_PROVIDER", "deterministic"),
        embedding_model=embedder.model_name,
        config_summary=_public_config(config),
    )

    store: QdrantChunkStore | None = None
    try:
        store = QdrantChunkStore(
            url=config.get("qdrant_url", "http://localhost:6333"),
            collection_name=report.collection_name,
            vector_size=embedder.vector_size,
            distance=config.get("distance", "Cosine"),
        )
        store.ensure_collection()
        report.worked.append("Ensured Qdrant collection and payload indexes.")

        updated_after = _resolve_updated_after(store, config)
        loader = ConfluenceMultiPageLoader(
            url=config["confluence_url"],
            username=config.get("confluence_username"),
            api_token=config.get("confluence_api_token"),
            parent_page_id=config.get("parent_page_id"),
            space_key=config.get("space_key"),
            page_ids=config.get("page_ids"),
            updated_after=updated_after,
            include_parent_page=config.get("include_parent_page", True),
            content_format=config.get("content_format", "view"),
        )

        discovered_ids = loader.discover_page_ids()
        report.pages_discovered = len(discovered_ids)
        report.worked.append(f"Discovered {len(discovered_ids)} page(s) for ingestion.")

        splitter = MarkdownDocumentTextSplitter(
            chunk_size=config.get("chunk_size", 1200),
            chunk_overlap=config.get("chunk_overlap", 150),
        )
        attachment_sample_dir = config.get(
            "attachment_sample_dir",
            "data/attachment_samples",
        )
        sampled_attachment_extensions: set[str] = set()

        for document in loader.lazy_load():
            page_id = str(document.metadata.get("page_id") or document.id)
            try:
                attachment_stats = loader.save_attachment_samples(
                    page_id,
                    attachment_sample_dir,
                    sampled_attachment_extensions,
                )
                report.attachments_discovered += attachment_stats["discovered"]
                report.attachment_samples_saved += attachment_stats["saved"]
                report.attachments_skipped += attachment_stats["skipped"]
                report.attachments_failed += attachment_stats["failed"]
                report.attachment_sample_extensions.extend(
                    attachment_stats["extensions"]
                )
                report.errors.extend(attachment_stats["errors"])
                if attachment_stats["saved"]:
                    report.added.append(
                        "Saved attachment sample(s) for page "
                        f"`{page_id}`: {', '.join(attachment_stats['extensions'])}."
                    )

                if not document.page_content.strip():
                    report.pages_skipped += 1
                    report.removed.append(f"Skipped empty page `{page_id}`.")
                    continue

                chunks = splitter.split_documents([document])
                chunks = [enrich_chunk_metadata(chunk) for chunk in chunks]
                report.chunks_created += len(chunks)

                deleted = store.delete_by_page_id(page_id)
                report.points_deleted += deleted
                if deleted:
                    report.removed.append(
                        f"Deleted `{deleted}` old point(s) for page `{page_id}`."
                    )

                vectors = embedder.embed_documents(
                    [chunk.page_content for chunk in chunks]
                )
                upserted = store.upsert_chunks(chunks, vectors)
                report.points_upserted += upserted
                report.pages_loaded += 1
                report.added.append(
                    f"Upserted `{upserted}` chunk point(s) for page `{page_id}`."
                )
            except Exception as error:
                report.pages_failed += 1
                report.failed.append(f"Failed to ingest page `{page_id}`: {error}")
                report.errors.append(traceback.format_exc(limit=3))

        if report.pages_loaded:
            report.worked.append(
                f"Loaded {report.pages_loaded} page(s) into `{report.collection_name}`."
            )
        if report.pages_failed:
            report.failed.append(
                f"{report.pages_failed} page(s) failed during ingestion."
            )

    except Exception as error:
        report.failed.append(f"Ingestion pipeline failed: {error}")
        report.errors.append(traceback.format_exc(limit=5))
    finally:
        changelog_dir = config.get("changelog_dir", "changelog")
        report.write_markdown(changelog_dir)

    return report


def _resolve_updated_after(store: QdrantChunkStore, config: dict[str, Any]) -> datetime:
    if config.get("updated_after"):
        return _parse_cli_datetime(config["updated_after"])

    if config.get("full_reload"):
        return DEFAULT_UPDATED_AFTER

    latest = store.get_max_source_updated_at()
    return latest or DEFAULT_UPDATED_AFTER


def _parse_cli_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _describe_scope(config: dict[str, Any]) -> str:
    if config.get("page_ids"):
        return f"page_ids={config['page_ids']}"
    if config.get("parent_page_id"):
        return f"parent_page_id={config['parent_page_id']}"
    if config.get("space_key"):
        return f"space_key={config['space_key']}"
    return "unspecified"


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    hidden_keys = {"confluence_api_token", "confluence_password"}
    return {
        key: value
        for key, value in config.items()
        if key not in hidden_keys and value is not None
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load Confluence pages, split, embed, and upsert chunks into Qdrant."
    )
    parser.add_argument("--confluence-url", default=os.getenv("CONFLUENCE_URL"))
    parser.add_argument("--confluence-username", default=os.getenv("CONFLUENCE_USERNAME"))
    parser.add_argument("--confluence-api-token", default=os.getenv("CONFLUENCE_API_TOKEN"))
    parser.add_argument("--parent-page-id", default=os.getenv("CONFLUENCE_PARENT_PAGE_ID"))
    parser.add_argument("--space-key", default=os.getenv("CONFLUENCE_SPACE_KEY"))
    parser.add_argument("--page-ids", default=os.getenv("CONFLUENCE_PAGE_IDS"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument(
        "--collection-name",
        default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
    )
    parser.add_argument(
        "--embedding-provider",
        default=os.getenv("EMBEDDING_PROVIDER", "deterministic"),
    )
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL"))
    parser.add_argument(
        "--embedding-vector-size",
        type=int,
        default=int(os.getenv("EMBEDDING_VECTOR_SIZE", "384")),
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--updated-after")
    parser.add_argument("--full-reload", action="store_true")
    parser.add_argument("--include-parent-page", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--content-format", default="view")
    parser.add_argument("--changelog-dir", default="changelog")
    parser.add_argument(
        "--attachment-sample-dir",
        default=os.getenv("ATTACHMENT_SAMPLE_DIR", "data/attachment_samples"),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.confluence_url:
        parser.error("`--confluence-url` or `CONFLUENCE_URL` is required.")

    page_ids = None
    if args.page_ids:
        page_ids = [item.strip() for item in args.page_ids.split(",") if item.strip()]

    config = {
        "confluence_url": args.confluence_url,
        "confluence_username": args.confluence_username or None,
        "confluence_api_token": args.confluence_api_token or None,
        "parent_page_id": args.parent_page_id or None,
        "space_key": args.space_key or None,
        "page_ids": page_ids,
        "qdrant_url": args.qdrant_url,
        "collection_name": args.collection_name,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "embedding_vector_size": args.embedding_vector_size,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "updated_after": args.updated_after,
        "full_reload": args.full_reload,
        "include_parent_page": args.include_parent_page,
        "content_format": args.content_format,
        "changelog_dir": args.changelog_dir,
        "attachment_sample_dir": args.attachment_sample_dir,
    }

    report = run_ingestion(config)
    print(
        "Ingestion finished: "
        f"loaded={report.pages_loaded}, failed={report.pages_failed}, "
        f"upserted={report.points_upserted}"
    )


if __name__ == "__main__":
    main()
