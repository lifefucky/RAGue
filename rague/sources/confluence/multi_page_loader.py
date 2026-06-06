"""Multi-page Confluence loader for ingestion pipelines."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
from typing import TYPE_CHECKING, Any

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from rague.sources.confluence.loader import (
    CONFLUENCE_SOURCE,
    DEFAULT_CONTENT_FORMAT,
    _compact_page_metadata,
    _create_confluence_client,
    _normalize_content_format,
    _normalize_page_ids,
    _page_body,
    confluence_html_to_markdown,
)

if TYPE_CHECKING:
    from atlassian import Confluence

DEFAULT_UPDATED_AFTER = datetime(1970, 1, 1, tzinfo=timezone.utc)
PAGE_BATCH_SIZE = 100
ATTACHMENT_BATCH_SIZE = 100


class ConfluenceMultiPageLoader(BaseLoader):
    """Discover and load multiple Confluence pages as LangChain `Document` objects.

    Scope (exactly one required unless `page_ids` is provided):
    - `parent_page_id`: recursively load the parent and all descendant pages.
    - `space_key`: discover pages in a Confluence space (optionally filtered by date).
    - `page_ids`: explicit list of page IDs for testing or targeted reloads.
    """

    def __init__(
        self,
        *,
        url: str,
        username: str | None = None,
        password: str | None = None,
        api_token: str | None = None,
        token: str | None = None,
        parent_page_id: str | int | None = None,
        space_key: str | None = None,
        page_ids: Sequence[str | int] | str | int | None = None,
        updated_after: datetime | None = None,
        include_parent_page: bool = True,
        content_format: str = DEFAULT_CONTENT_FORMAT,
        confluence: Confluence | None = None,
        confluence_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.parent_page_id = str(parent_page_id) if parent_page_id is not None else None
        self.space_key = space_key
        self.page_ids = _normalize_page_ids(page_ids)
        self.updated_after = updated_after or DEFAULT_UPDATED_AFTER
        self.include_parent_page = include_parent_page
        self.content_format = _normalize_content_format(content_format)
        self._confluence = confluence or _create_confluence_client(
            url=self.url,
            username=username,
            password=password or api_token,
            token=token,
            confluence_kwargs=confluence_kwargs,
        )

        if not self.page_ids and not self.parent_page_id and not self.space_key:
            raise ValueError(
                "Provide one of: `parent_page_id`, `space_key`, or `page_ids`."
            )

    def discover_page_ids(self) -> list[str]:
        """Return page IDs matching the configured scope and update filter."""
        if self.page_ids:
            candidates = list(dict.fromkeys(self.page_ids))
        elif self.parent_page_id:
            candidates = self._discover_descendant_page_ids(self.parent_page_id)
            if self.include_parent_page:
                candidates = [self.parent_page_id, *candidates]
            candidates = list(dict.fromkeys(candidates))
        elif self.space_key:
            candidates = self._discover_space_page_ids(self.space_key)
        else:
            candidates = []

        return [
            page_id
            for page_id in candidates
            if self._is_updated_after(page_id, self.updated_after)
        ]

    def load_page_ids(self, page_ids: Sequence[str | int]) -> Iterator[Document]:
        """Yield documents for explicit page IDs without running discovery."""
        for page_id in page_ids:
            page = self._get_page(str(page_id), include_body=True)
            markdown = confluence_html_to_markdown(
                _page_body(page, self.content_format)
            )
            metadata = self._metadata_for_page(page)

            yield Document(
                id=metadata["page_id"],
                page_content=markdown,
                metadata=metadata,
            )

    def lazy_load(self) -> Iterator[Document]:
        """Yield one `Document` per discovered Confluence page."""
        yield from self.load_page_ids(self.discover_page_ids())

    def save_attachment_samples(
        self,
        page_id: str,
        output_dir: str | Path,
        seen_extensions: set[str],
    ) -> dict[str, Any]:
        """Save at most one attachment sample per file extension.

        Attachments are intentionally not converted to `Document` objects here and
        are not indexed. Samples are local fixtures for future converter work.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stats: dict[str, Any] = {
            "discovered": 0,
            "saved": 0,
            "skipped": 0,
            "failed": 0,
            "extensions": [],
            "errors": [],
        }

        try:
            for attachment in self._iter_attachments(page_id):
                stats["discovered"] += 1
                filename = _attachment_filename(attachment)
                extension = _attachment_extension(filename)

                if extension in seen_extensions:
                    stats["skipped"] += 1
                    continue

                try:
                    content = self._download_attachment(attachment)
                    attachment_id = str(attachment.get("id", "unknown"))
                    sample_path = output_path / _sample_filename(
                        extension,
                        attachment_id,
                        filename,
                    )
                    sample_path.write_bytes(content)
                except Exception as error:
                    stats["failed"] += 1
                    stats["errors"].append(f"{filename}: {error}")
                    continue

                seen_extensions.add(extension)
                stats["saved"] += 1
                stats["extensions"].append(extension)
        except Exception as error:
            stats["failed"] += 1
            stats["errors"].append(f"page {page_id} attachments: {error}")

        return stats

    def _iter_attachments(self, page_id: str) -> Iterator[dict[str, Any]]:
        start = 0
        while True:
            response = self._confluence.get_attachments_from_content(
                page_id,
                start=start,
                limit=ATTACHMENT_BATCH_SIZE,
                expand="version",
            )
            results = _extract_results(response)
            if not results:
                break

            yield from results

            if len(results) < ATTACHMENT_BATCH_SIZE:
                break
            start += ATTACHMENT_BATCH_SIZE

    def _download_attachment(self, attachment: dict[str, Any]) -> bytes:
        download_url = _attachment_download_url(attachment, self.url)

        session = (
            getattr(self._confluence, "session", None)
            or getattr(self._confluence, "_session", None)
        )
        if session is not None:
            response = session.get(download_url)
            response.raise_for_status()
            return response.content

        request = getattr(self._confluence, "request", None)
        if request is not None:
            response = request(
                path=download_url,
                absolute=True,
                advanced_mode=True,
            )
            content = getattr(response, "content", response)
            if isinstance(content, bytes):
                return content

        message = "Confluence client does not expose a supported attachment download API."
        raise RuntimeError(message)

    def _discover_descendant_page_ids(self, root_page_id: str) -> list[str]:
        discovered: list[str] = []
        queue = [root_page_id]

        while queue:
            current_id = queue.pop(0)
            start = 0
            while True:
                response = self._confluence.get_page_child_by_type(
                    current_id,
                    type="page",
                    start=start,
                    limit=PAGE_BATCH_SIZE,
                    expand="version",
                )
                results = _extract_results(response)
                if not results:
                    break

                for child in results:
                    child_id = str(child["id"])
                    if child_id not in discovered:
                        discovered.append(child_id)
                        queue.append(child_id)

                if len(results) < PAGE_BATCH_SIZE:
                    break
                start += PAGE_BATCH_SIZE

        return discovered

    def _discover_space_page_ids(self, space_key: str) -> list[str]:
        discovered: list[str] = []
        start = 0
        updated_after_iso = _format_cql_datetime(self.updated_after)

        while True:
            cql = (
                f'space = "{space_key}" and type = page '
                f'and lastmodified >= "{updated_after_iso}" order by lastmodified asc'
            )
            response = self._confluence.cql(cql, start=start, limit=PAGE_BATCH_SIZE)
            results = _extract_cql_page_ids(response)
            if not results:
                break

            for page_id in results:
                if page_id not in discovered:
                    discovered.append(page_id)

            if len(results) < PAGE_BATCH_SIZE:
                break
            start += PAGE_BATCH_SIZE

        return discovered

    def _is_updated_after(self, page_id: str, updated_after: datetime) -> bool:
        if self.page_ids and not self.parent_page_id and not self.space_key:
            return True

        if self.space_key and not self.page_ids and not self.parent_page_id:
            return True

        page = self._get_page(page_id, include_body=False)
        page_updated_at = _parse_confluence_datetime(
            page.get("version", {}).get("when")
        )
        if page_updated_at is None:
            return True
        return page_updated_at > updated_after

    def _get_page(self, page_id: str, *, include_body: bool) -> dict[str, Any]:
        expand_parts = ["version", "space", "ancestors"]
        if include_body:
            expand_parts.append(f"body.{self.content_format}")

        return self._confluence.get_page_by_id(
            page_id,
            expand=",".join(expand_parts),
        )

    def _metadata_for_page(self, page: dict[str, Any]) -> dict[str, Any]:
        compact_page = _compact_page_metadata(page, self.url)
        page_id = str(compact_page["id"])
        ancestors = [
            _compact_page_metadata(ancestor, self.url)
            for ancestor in page.get("ancestors", [])
        ]
        source_updated_at = page.get("version", {}).get("when")
        path = _build_page_path(ancestors, compact_page["title"])
        parent_page_id = ancestors[-1]["id"] if ancestors else None
        ingested_at = datetime.now(timezone.utc).isoformat()

        return {
            **compact_page,
            "source_type": CONFLUENCE_SOURCE,
            "document_type": "page",
            "document_id": f"confluence:page:{page_id}",
            "page_id": page_id,
            "page_version": compact_page["version"],
            "source_updated_at": source_updated_at,
            "ingested_at": ingested_at,
            "parent_page_id": parent_page_id,
            "parent_page_ids": [ancestor["id"] for ancestor in ancestors],
            "parent_titles": [ancestor["title"] for ancestor in ancestors],
            "ancestors": ancestors,
            "path": path,
            "content_format": self.content_format,
            "is_current": True,
        }


def _extract_results(response: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        if "results" in response:
            return response["results"]
        if "page" in response:
            pages = response["page"]
            return pages.get("results", pages) if isinstance(pages, dict) else pages
    return []


def _extract_cql_page_ids(response: dict[str, Any]) -> list[str]:
    page_ids: list[str] = []
    for item in response.get("results", []):
        content = item.get("content") or item
        if content.get("id"):
            page_ids.append(str(content["id"]))
    return page_ids


def _build_page_path(ancestors: list[dict[str, Any]], title: str) -> str:
    parts = [ancestor["title"] for ancestor in ancestors if ancestor.get("title")]
    parts.append(title)
    return "/".join(parts)


def _attachment_filename(attachment: dict[str, Any]) -> str:
    return (
        attachment.get("title")
        or attachment.get("metadata", {}).get("mediaType")
        or str(attachment.get("id", "attachment"))
    )


def _attachment_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower().lstrip(".")
    return extension or "no_extension"


def _attachment_download_url(attachment: dict[str, Any], base_url: str) -> str:
    links = attachment.get("_links", {})
    download_path = links.get("download")
    if not download_path:
        message = f"Attachment `{_attachment_filename(attachment)}` has no download link."
        raise ValueError(message)
    return urljoin(base_url.rstrip("/") + "/", download_path.lstrip("/"))


def _sample_filename(extension: str, attachment_id: str, filename: str) -> str:
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in filename
    ).strip("._")
    stem = Path(safe_name).stem or "attachment"
    return f"{extension}_{attachment_id}_{stem}.{extension}"


def _format_cql_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _parse_confluence_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
