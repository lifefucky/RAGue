"""LangChain loader for corporate Confluence pages."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

if TYPE_CHECKING:
    from atlassian import Confluence


DEFAULT_CONTENT_FORMAT = "view"
CONFLUENCE_SOURCE = "confluence"


class CorporateConfluenceLoader(BaseLoader):
    """Load corporate Confluence pages as LangChain `Document` objects.

    `page_ids`/`topic_page_ids` are loaded as content. `parent_page_ids` are
    fetched only for metadata enrichment and attached to every loaded document.
    """

    def __init__(
        self,
        *,
        url: str,
        username: str | None = None,
        password: str | None = None,
        api_token: str | None = None,
        token: str | None = None,
        page_ids: Sequence[str | int] | str | int | None = None,
        topic_page_ids: Sequence[str | int] | str | int | None = None,
        parent_page_ids: Sequence[str | int] | str | int | None = None,
        content_format: str = DEFAULT_CONTENT_FORMAT,
        confluence: Confluence | None = None,
        confluence_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.page_ids = _normalize_page_ids(page_ids or topic_page_ids)
        if not self.page_ids:
            raise ValueError("Pass at least one page ID via `page_ids` or `topic_page_ids`.")

        self.parent_page_ids = _normalize_page_ids(parent_page_ids)
        self.content_format = _normalize_content_format(content_format)
        self._confluence = confluence or _create_confluence_client(
            url=self.url,
            username=username,
            password=password or api_token,
            token=token,
            confluence_kwargs=confluence_kwargs,
        )
        self._parent_pages: list[dict[str, Any]] | None = None

    def lazy_load(self) -> Iterator[Document]:
        """Yield one `Document` per Confluence page."""
        parents = self._load_parent_pages()

        for page_id in self.page_ids:
            page = self._get_page(page_id, include_body=True)
            markdown = confluence_html_to_markdown(
                _page_body(page, self.content_format)
            )
            metadata = self._metadata_for_page(page, parents)

            yield Document(
                id=metadata["id"],
                page_content=markdown,
                metadata=metadata,
            )

    def save_markdown(
        self,
        output_dir: str | Path,
        *,
        documents: Iterable[Document] | None = None,
    ) -> list[Path]:
        """Save loaded pages as Markdown files with YAML frontmatter."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        written_files: list[Path] = []
        for document in documents or self.lazy_load():
            page_id = str(document.metadata["id"])
            file_path = output_path / f"{page_id}.md"
            file_path.write_text(document_to_markdown(document), encoding="utf-8")
            written_files.append(file_path)

        return written_files

    def _load_parent_pages(self) -> list[dict[str, Any]]:
        if self._parent_pages is None:
            self._parent_pages = [
                _compact_page_metadata(self._get_page(page_id, include_body=False), self.url)
                for page_id in self.parent_page_ids
            ]
        return self._parent_pages

    def _get_page(self, page_id: str, *, include_body: bool) -> dict[str, Any]:
        expand_parts = ["version", "space", "ancestors"]
        if include_body:
            expand_parts.append(f"body.{self.content_format}")

        return self._confluence.get_page_by_id(
            page_id,
            expand=",".join(expand_parts),
        )

    def _metadata_for_page(
        self,
        page: dict[str, Any],
        parents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        compact_page = _compact_page_metadata(page, self.url)
        ancestors = [
            _compact_page_metadata(ancestor, self.url)
            for ancestor in page.get("ancestors", [])
        ]

        return {
            **compact_page,
            "source_type": CONFLUENCE_SOURCE,
            "content_format": self.content_format,
            "parents": parents,
            "parent_page_ids": [parent["id"] for parent in parents],
            "parent_titles": [parent["title"] for parent in parents],
            "ancestors": ancestors,
        }


def confluence_html_to_markdown(raw_html: str) -> str:
    """Convert Confluence view/storage HTML to Markdown."""
    if not raw_html:
        return ""

    soup = _parse_html(raw_html)

    for link in soup.find_all("ac:link"):
        page_ref = link.find("ri:page")
        user_ref = link.find("ri:user")

        if page_ref and page_ref.get("ri:content-title"):
            title = page_ref["ri:content-title"]
            anchor = title.replace(" ", "-").lower()
            link.replace_with(f"[{title}](#{anchor})")
        elif user_ref and user_ref.get("ri:display-name"):
            link.replace_with(f"@{user_ref['ri:display-name']}")
        else:
            link.replace_with(link.get_text())

    for macro in soup.find_all("ac:structured-macro"):
        name = macro.get("ac:name")
        if name == "code":
            language = macro.find("ac:parameter", {"ac:name": "language"})
            code_body = macro.find("ac:plain-text-body")
            if code_body:
                macro.replace_with(
                    f"```{language.get_text() if language else ''}\n"
                    f"{code_body.get_text()}\n```"
                )
            else:
                macro.decompose()
        elif name in ("info", "warning", "note", "tip"):
            macro.replace_with(f"> {macro.get_text().strip()}\n")

    for image in soup.find_all("ac:image"):
        attachment = image.find("ri:attachment")
        if attachment and attachment.get("ri:filename"):
            filename = attachment["ri:filename"]
            image.replace_with(f"![{filename}](attachments/{filename})")

    markdownify_html = _get_markdownify()
    return markdownify_html(
        str(soup),
        heading_style="atx",
        strip=["script", "style"],
    ).strip()


def document_to_markdown(document: Document) -> str:
    """Render a LangChain document to a Markdown file with metadata frontmatter."""
    metadata = document.metadata
    frontmatter_keys = (
        "title",
        "id",
        "version",
        "source",
        "space",
        "parents",
        "parent_page_ids",
        "parent_titles",
        "ancestors",
    )
    frontmatter = [
        f"{key}: {_yaml_value(metadata[key])}"
        for key in frontmatter_keys
        if metadata.get(key) not in (None, "", [], {})
    ]

    return "---\n" + "\n".join(frontmatter) + f"\n---\n\n{document.page_content}\n"


def _parse_html(raw_html: str) -> Any:
    try:
        from bs4 import BeautifulSoup, FeatureNotFound
    except ImportError as error:
        message = (
            "Packages `beautifulsoup4` and `markdownify` are required to convert "
            "Confluence HTML to Markdown."
        )
        raise ImportError(message) from error

    try:
        return BeautifulSoup(raw_html, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(raw_html, "html.parser")


def _get_markdownify() -> Any:
    try:
        from markdownify import markdownify as markdownify_html
    except ImportError as error:
        message = (
            "Package `markdownify` is required to convert Confluence HTML to Markdown."
        )
        raise ImportError(message) from error

    return markdownify_html


def _page_body(page: dict[str, Any], content_format: str) -> str:
    return page.get("body", {}).get(content_format, {}).get("value", "")


def _compact_page_metadata(page: dict[str, Any], base_url: str) -> dict[str, Any]:
    page_id = str(page.get("id", ""))
    space = page.get("space", {})

    return {
        "id": page_id,
        "title": page.get("title", "unknown"),
        "version": page.get("version", {}).get("number", "unknown"),
        "space": space.get("key"),
        "source": f"{base_url}/pages/viewpage.action?pageId={page_id}",
    }


def _create_confluence_client(
    *,
    url: str,
    username: str | None,
    password: str | None,
    token: str | None,
    confluence_kwargs: dict[str, Any] | None,
) -> Confluence:
    try:
        from atlassian import Confluence
    except ImportError as error:
        message = (
            "Package `atlassian-python-api` is required to load Confluence pages. "
            "Install it in the project environment before using "
            "`CorporateConfluenceLoader`."
        )
        raise ImportError(message) from error

    return Confluence(
        url=url,
        username=username,
        password=password,
        token=token,
        **(confluence_kwargs or {}),
    )


def _normalize_page_ids(
    page_ids: Sequence[str | int] | str | int | None,
) -> list[str]:
    if page_ids is None:
        return []
    if isinstance(page_ids, str | int):
        return [str(page_ids)]
    return [str(page_id) for page_id in page_ids]


def _normalize_content_format(content_format: str) -> str:
    return content_format.removeprefix("body.")


def _yaml_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
