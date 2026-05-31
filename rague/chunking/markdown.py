"""Markdown-aware LangChain text splitters."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    TextSplitter,
)

DEFAULT_HEADERS_TO_SPLIT_ON = (
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
    ("####", "header_4"),
    ("#####", "header_5"),
    ("######", "header_6"),
)


class MarkdownDocumentTextSplitter(TextSplitter):
    """Split Markdown `Document` objects while preserving header context.

    The splitter is intended for documents already normalized to Markdown, such as
    pages produced by `CorporateConfluenceLoader`.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        headers_to_split_on: Sequence[tuple[str, str]] = DEFAULT_HEADERS_TO_SPLIT_ON,
        strip_headers: bool = False,
        add_start_index: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
            **kwargs,
        )
        self.headers_to_split_on = list(headers_to_split_on)
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=strip_headers,
        )
        self._chunk_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
            **kwargs,
        )

    def split_text(self, text: str) -> list[str]:
        """Split Markdown text and return chunk contents only."""
        return [
            document.page_content
            for document in self._split_document(Document(page_content=text))
        ]

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Split LangChain documents and keep source metadata on each chunk."""
        chunks: list[Document] = []
        for document in documents:
            chunks.extend(self._split_document(document))
        return chunks

    def _split_document(self, document: Document) -> list[Document]:
        sections = self._header_splitter.split_text(document.page_content)
        if not sections:
            sections = [Document(page_content=document.page_content, metadata={})]

        chunks: list[Document] = []
        for section_index, section in enumerate(sections):
            metadata = copy.deepcopy(document.metadata)
            metadata.update(section.metadata)
            metadata["section_index"] = section_index

            section_chunks = self._chunk_splitter.create_documents(
                [section.page_content],
                metadatas=[metadata],
            )
            for chunk in section_chunks:
                chunk.metadata["chunk_index"] = len(chunks)
                if document.id:
                    chunk.id = f"{document.id}:{len(chunks)}"
                chunks.append(chunk)

        return chunks
