"""Markdown-aware splitter for text, tables, and code fences."""

from __future__ import annotations

import copy
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
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

DEFAULT_VALIDATION_MARKDOWN = Path("docs/confluence/100000001.md")
DEFAULT_DEBEZIUM_MARKDOWN = Path("docs/confluence/131302699.md")

_ATX_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_NUMBERED_STEP_RE = re.compile(r"^(?:#{1,6}\s+)?\d+\.\s+.+")
_BOLD_CAPTION_RE = re.compile(r"^\*\*[^*]+\*\*")
_CONFLUENCE_EXPAND_TEXT_RE = re.compile(r"\s*Развернуть\s+исходный\s+код", re.IGNORECASE)
_SQL_SECTION_COMMENT_RE = re.compile(r"^--\s*(?:\d+\.|[\wА-Яа-я])")
_SQL_TRANSACTION_RE = re.compile(r"^(BEGIN|COMMIT|ROLLBACK);?\s*$", re.IGNORECASE)
_PYTHON_DEF_RE = re.compile(r"^(async\s+def|def|class)\s+")
_PYTHON_DIVIDER_RE = re.compile(r"^#\s*-{3,}")
_SQL_MARKERS = ("SELECT", "INSERT", "CREATE", "ALTER", "GRANT", "DROP", "BEGIN", "COMMIT")

_SQL_OPERATION_RE = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|GRANT|BEGIN|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)
_SQL_ENTITY_PATTERNS = (
    re.compile(r"\bFROM\s+([a-zA-Z_][\w.]*)", re.IGNORECASE),
    re.compile(r"\bJOIN\s+([a-zA-Z_][\w.]*)", re.IGNORECASE),
    re.compile(r"\bINTO\s+([a-zA-Z_][\w.]*)", re.IGNORECASE),
    re.compile(
        r"\bTABLE\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?([a-zA-Z_][\w.]*)",
        re.IGNORECASE,
    ),
    re.compile(r"\bON\s+TABLE\s+([a-zA-Z_][\w.]*)", re.IGNORECASE),
    re.compile(r"\bPUBLICATION\s+([a-zA-Z_][\w.]*)", re.IGNORECASE),
    re.compile(r"\bFUNCTION\s+([a-zA-Z_][\w.]*)", re.IGNORECASE),
)


@dataclass(frozen=True)
class _TableSegment:
    prefix: str
    header_sep: tuple[str, str]
    data_rows: tuple[str, ...]


@dataclass(frozen=True)
class _TextSegment:
    text: str


@dataclass(frozen=True)
class _CodeSegment:
    prefix: str
    language: str
    code: str


@dataclass
class _SectionChunk:
    content: str
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith("|")
        and stripped.endswith("|")
        and stripped.count("|") >= 2
    )


def _is_separator_row(line: str) -> bool:
    stripped = line.strip()
    if not _is_table_line(stripped):
        return False

    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return bool(cells) and all(cell and set(cell) <= {"-", ":", " "} for cell in cells)


def _parse_table_block(lines: Sequence[str]) -> tuple[tuple[str, str], tuple[str, ...]] | None:
    if len(lines) < 2 or not _is_separator_row(lines[1]):
        return None

    data_rows = tuple(line for line in lines[2:] if line.strip())
    if not data_rows:
        return None

    return (lines[0], lines[1]), data_rows


def _extract_atx_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if _ATX_HEADING_RE.match(stripped):
            return stripped
    return None


def _extract_last_numbered_step(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if _NUMBERED_STEP_RE.match(stripped):
            return stripped
    return None


def _extract_last_bold_caption(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if _BOLD_CAPTION_RE.match(stripped):
            return stripped
    return None


def _clean_confluence_expand_text(text: str) -> str:
    return _CONFLUENCE_EXPAND_TEXT_RE.sub("", text).strip()


def _build_code_context(prefix: str, section_heading: str | None) -> str:
    parts: list[str] = []
    heading = section_heading or _extract_atx_heading(prefix)
    if heading:
        parts.append(heading.strip())

    step = _extract_last_numbered_step(prefix)
    if step and step not in parts:
        parts.append(step)

    caption = _extract_last_bold_caption(prefix)
    if caption and caption not in parts:
        parts.append(caption)

    return "\n\n".join(parts)


def _detect_code_language(declared: str, code: str, prefix: str) -> str:
    if declared.strip():
        return declared.strip().lower()

    if "def " in code or "import " in code or "async def " in code:
        return "python"

    stripped = code.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"

    upper_code = code.upper()
    if any(marker in upper_code for marker in _SQL_MARKERS):
        return "sql"

    if "SQL" in prefix.upper():
        return "sql"

    return ""


def _dedupe_ordered(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _extract_sql_operations(code: str) -> list[str]:
    return _dedupe_ordered(match.group(1).upper() for match in _SQL_OPERATION_RE.finditer(code))


def _extract_sql_entities(code: str) -> list[str]:
    entities: list[str] = []
    for pattern in _SQL_ENTITY_PATTERNS:
        for match in pattern.finditer(code):
            entities.append(match.group(1))
    return _dedupe_ordered(entities)


def _extract_config_entities(code: str) -> list[str]:
    return _dedupe_ordered(
        match.group(1)
        for match in re.finditer(r"^(\w+)\s*=", code, flags=re.MULTILINE)
    )


def _summarize_sql_code(code: str) -> tuple[list[str], list[str]]:
    operations = _extract_sql_operations(code)
    entities = _extract_sql_entities(code)
    if operations or entities:
        return operations, entities

    config_entities = _extract_config_entities(code)
    if config_entities:
        return ["CONFIG"], config_entities

    return operations, entities


def _summarize_json_code(code: str) -> tuple[list[str], list[str]]:
    keys = _dedupe_ordered(
        match.group(1) for match in re.finditer(r'"(\w+)"\s*:', code)
    )
    return ["JSON"], keys


def _build_code_ref(document_metadata: dict[str, Any], code_block_index: int) -> str:
    page_id = str(document_metadata.get("page_id") or document_metadata.get("id", "unknown"))
    page_version = document_metadata.get("page_version") or document_metadata.get(
        "version",
        "unknown",
    )
    document_id = document_metadata.get("document_id") or f"confluence:page:{page_id}"
    return f"{document_id}:v{page_version}:code:{code_block_index}"


def _render_code_summary(
    *,
    context: str,
    language: str,
    operations: Sequence[str],
    entities: Sequence[str],
    code_ref: str,
    caption: str | None,
) -> str:
    lines: list[str] = []
    if context:
        lines.extend(context.splitlines())
        lines.append("")

    if caption:
        lines.append(f"Блок: {caption}")

    lines.append(f"Тип: {language.upper() if language else 'CODE'}")
    if operations:
        lines.append(f"Операции: {', '.join(operations)}")
    if entities:
        lines.append(f"Сущности: {', '.join(entities)}")
    lines.append(f"Полный код: {code_ref}")
    return "\n".join(lines)


def _merge_code_sections(sections: list[list[str]], max_size: int) -> list[list[str]]:
    if not sections:
        return []

    merged: list[list[str]] = []
    current: list[str] = []

    for section in sections:
        candidate = current + section
        candidate_text = "\n".join(candidate)
        if not current or len(candidate_text) <= max_size:
            current = candidate
            continue

        merged.append(current)
        current = section

    if current:
        merged.append(current)

    return merged


def _split_code_by_lines(code: str, max_size: int) -> list[str]:
    if len(code) <= max_size:
        return [code]

    lines = code.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_size:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
            continue

        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks


def _split_sql_code(code: str, max_size: int) -> list[str]:
    if len(code) <= max_size:
        return [code]

    lines = code.splitlines()
    sections: list[list[str]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            sections.append(current)
            current = []

    for index, line in enumerate(lines):
        stripped = line.strip()

        if (
            current
            and stripped.startswith("--")
            and _SQL_SECTION_COMMENT_RE.match(stripped)
        ):
            flush()

        current.append(line)

        if _SQL_TRANSACTION_RE.match(stripped):
            flush()
            continue

        if (
            stripped.endswith(";")
            and index + 1 < len(lines)
            and not lines[index + 1].strip()
        ):
            flush()

    flush()

    if not sections:
        return _split_code_by_lines(code, max_size)

    merged_sections = _merge_code_sections(sections, max_size)
    result: list[str] = []
    for section in merged_sections:
        section_text = "\n".join(section).strip()
        if not section_text:
            continue
        if len(section_text) <= max_size:
            result.append(section_text)
        else:
            result.extend(_split_code_by_lines(section_text, max_size))

    return result or [code]


def _split_python_code(code: str, max_size: int) -> list[str]:
    if len(code) <= max_size:
        return [code]

    lines = code.splitlines()
    sections: list[list[str]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            sections.append(current)
            current = []

    for line in lines:
        stripped = line.strip()
        if current and (
            _PYTHON_DEF_RE.match(stripped) or _PYTHON_DIVIDER_RE.match(stripped)
        ):
            flush()
        current.append(line)

    flush()

    if not sections:
        return _split_code_by_lines(code, max_size)

    merged_sections = _merge_code_sections(sections, max_size)
    result: list[str] = []
    for section in merged_sections:
        section_text = "\n".join(section).strip()
        if not section_text:
            continue
        if len(section_text) <= max_size:
            result.append(section_text)
        else:
            result.extend(_split_code_by_lines(section_text, max_size))

    return result or [code]


def _split_generic_code(code: str, max_size: int) -> list[str]:
    if len(code) <= max_size:
        return [code]

    lines = code.splitlines()
    sections: list[list[str]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            sections.append(current)
            current = []

    for line in lines:
        if current and not line.strip():
            flush()
        current.append(line)

    flush()

    if len(sections) <= 1:
        return _split_code_by_lines(code, max_size)

    merged_sections = _merge_code_sections(sections, max_size)
    result: list[str] = []
    for section in merged_sections:
        section_text = "\n".join(section).strip()
        if not section_text:
            continue
        if len(section_text) <= max_size:
            result.append(section_text)
        else:
            result.extend(_split_code_by_lines(section_text, max_size))

    return result or [code]


def _split_code_body(code: str, language: str, max_size: int) -> list[str]:
    if language == "sql":
        return _split_sql_code(code, max_size)
    if language == "python":
        return _split_python_code(code, max_size)
    return _split_generic_code(code, max_size)


def _render_code_fence(language: str, code: str) -> str:
    if language:
        return f"```{language}\n{code}\n```"
    return f"```\n{code}\n```"


def _segment_by_blocks(text: str) -> list[_TableSegment | _TextSegment | _CodeSegment]:
    segments: list[_TableSegment | _TextSegment | _CodeSegment] = []
    lines = text.splitlines()
    buffer: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            language = stripped[3:].strip()
            code_lines: list[str] = []
            index += 1
            while index < len(lines):
                if lines[index].strip().startswith("```"):
                    index += 1
                    break
                code_lines.append(lines[index])
                index += 1

            segments.append(
                _CodeSegment(
                    prefix=_clean_confluence_expand_text("\n".join(buffer)),
                    language=language,
                    code="\n".join(code_lines),
                )
            )
            buffer = []
            continue

        if _is_table_line(line):
            table_lines: list[str] = []
            while index < len(lines) and _is_table_line(lines[index]):
                table_lines.append(lines[index])
                index += 1

            parsed = _parse_table_block(table_lines)
            if parsed is None:
                buffer.extend(table_lines)
                continue

            header_sep, data_rows = parsed
            segments.append(
                _TableSegment(
                    prefix=_clean_confluence_expand_text("\n".join(buffer)),
                    header_sep=header_sep,
                    data_rows=data_rows,
                )
            )
            buffer = []
            continue

        buffer.append(line)
        index += 1

    trailing_text = _clean_confluence_expand_text("\n".join(buffer))
    if trailing_text:
        segments.append(_TextSegment(text=trailing_text))

    return segments


class MarkdownDocumentTextSplitter(TextSplitter):
    """Split Markdown with one chunk per table row and code-fence aware chunking."""

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
        self.chunk_size = chunk_size
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
        code_block_counter = 0

        for section_index, section in enumerate(sections):
            metadata = copy.deepcopy(document.metadata)
            metadata.update(section.metadata)
            metadata["section_index"] = section_index

            section_chunks, code_block_counter = self._split_section(
                section.page_content,
                code_block_counter,
                document_metadata=metadata,
            )
            for section_chunk in section_chunks:
                chunk_metadata = copy.deepcopy(metadata)
                chunk_metadata["chunk_index"] = len(chunks)
                chunk_metadata.update(section_chunk.extra_metadata)

                chunk = Document(
                    page_content=section_chunk.content,
                    metadata=chunk_metadata,
                )
                if document.id:
                    chunk.id = f"{document.id}:{len(chunks)}"
                chunks.append(chunk)

        return chunks

    def _split_section(
        self,
        text: str,
        code_block_counter: int = 0,
        *,
        document_metadata: dict[str, Any] | None = None,
    ) -> tuple[list[_SectionChunk], int]:
        document_metadata = document_metadata or {}
        segments = _segment_by_blocks(text)
        has_special_blocks = any(
            isinstance(segment, (_TableSegment, _CodeSegment)) for segment in segments
        )

        if not has_special_blocks:
            return (
                [
                    _SectionChunk(
                        content=document.page_content,
                        extra_metadata={"chunk_type": "text"},
                    )
                    for document in self._chunk_splitter.create_documents([text])
                ],
                code_block_counter,
            )

        chunks: list[_SectionChunk] = []
        section_heading = _extract_atx_heading(text)

        for segment in segments:
            if isinstance(segment, _TableSegment):
                if section_heading is None:
                    section_heading = _extract_atx_heading(segment.prefix)

                for row_index, data_row in enumerate(segment.data_rows):
                    mini_table = "\n".join((*segment.header_sep, data_row))
                    if segment.prefix:
                        chunk = f"{segment.prefix}\n\n{mini_table}"
                    elif section_heading:
                        chunk = f"{section_heading}\n\n{mini_table}"
                    else:
                        chunk = mini_table

                    chunks.append(
                        _SectionChunk(
                            content=chunk.strip(),
                            extra_metadata={
                                "chunk_type": "table_row",
                                "table_row_index": row_index,
                            },
                        )
                    )
                continue

            if isinstance(segment, _CodeSegment):
                if section_heading is None:
                    section_heading = _extract_atx_heading(segment.prefix)

                context = _build_code_context(segment.prefix, section_heading)
                language = _detect_code_language(
                    segment.language,
                    segment.code,
                    segment.prefix,
                )
                local_heading = _extract_last_numbered_step(segment.prefix)
                caption = _extract_last_bold_caption(segment.prefix)
                block_index = code_block_counter
                code_block_counter += 1

                if language in {"sql", "json"}:
                    if language == "json":
                        operations, entities = _summarize_json_code(segment.code)
                    else:
                        operations, entities = _summarize_sql_code(segment.code)
                    code_ref = _build_code_ref(document_metadata, block_index)
                    content = _render_code_summary(
                        context=context,
                        language=language,
                        operations=operations,
                        entities=entities,
                        code_ref=code_ref,
                        caption=caption,
                    )
                    chunks.append(
                        _SectionChunk(
                            content=content.strip(),
                            extra_metadata={
                                "chunk_type": "code_summary",
                                "code_language": language,
                                "code_block_index": block_index,
                                "code_ref": code_ref,
                                "code_operations": operations,
                                "code_entities": entities,
                                "raw_code": segment.code,
                                "local_heading": local_heading,
                                "caption": caption,
                                "chunk_id": code_ref,
                            },
                        )
                    )
                    continue

                context_budget = len(context) + 20 if context else 0
                code_budget = max(self.chunk_size - context_budget, self.chunk_size // 2)
                code_fragments = _split_code_body(segment.code, language, code_budget)

                for code_chunk_index, fragment in enumerate(code_fragments):
                    fence = _render_code_fence(language, fragment)
                    content = f"{context}\n\n{fence}" if context else fence

                    chunks.append(
                        _SectionChunk(
                            content=content.strip(),
                            extra_metadata={
                                "chunk_type": "code",
                                "code_language": language or None,
                                "code_block_index": block_index,
                                "code_chunk_index": code_chunk_index,
                                "local_heading": local_heading,
                                "caption": caption,
                            },
                        )
                    )
                continue

            trailing_text = segment.text.strip()
            if not trailing_text:
                continue

            heading_prefix = ""
            if section_heading and section_heading not in trailing_text:
                heading_prefix = f"{section_heading}\n\n"

            for piece in self._chunk_splitter.split_text(trailing_text):
                chunks.append(
                    _SectionChunk(
                        content=f"{heading_prefix}{piece}".strip(),
                        extra_metadata={"chunk_type": "text"},
                    )
                )

        return chunks, code_block_counter


def load_markdown_body(path: str | Path) -> str:
    """Load a markdown file and return body text without YAML frontmatter."""
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("---"):
        return text

    end = text.find("\n---", 3)
    if end == -1:
        return text

    return text[end + 4 :].lstrip("\n")


def load_markdown_source_metadata(path: str | Path) -> dict[str, Any]:
    """Load page identifiers from YAML frontmatter when present."""
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}

    end = text.find("\n---", 3)
    if end == -1:
        return {}

    metadata: dict[str, Any] = {}
    for line in text[3:end].splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        cleaned = value.strip().strip('"')
        metadata[key.strip()] = cleaned

    page_id = metadata.get("id")
    if not page_id:
        return metadata

    page_id_str = str(page_id)
    page_version = metadata.get("version", "unknown")
    document_id = f"confluence:page:{page_id_str}"
    return {
        **metadata,
        "page_id": page_id_str,
        "id": page_id_str,
        "page_version": page_version,
        "version": page_version,
        "document_id": document_id,
    }


def split_markdown_file(
    path: str | Path = DEFAULT_VALIDATION_MARKDOWN,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    **splitter_kwargs: Any,
) -> list[Document]:
    """Split a markdown file from disk and return LangChain chunk documents."""
    markdown_path = Path(path)
    body = load_markdown_body(markdown_path)
    source_metadata = load_markdown_source_metadata(markdown_path)
    source_metadata["source"] = str(markdown_path.resolve())
    splitter = MarkdownDocumentTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **splitter_kwargs,
    )
    source = Document(
        page_content=body,
        metadata=source_metadata,
    )
    return splitter.split_documents([source])


def print_chunks_from_file(
    path: str | Path = DEFAULT_VALIDATION_MARKDOWN,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    **splitter_kwargs: Any,
) -> list[Document]:
    """Split a markdown file and print all chunks for manual validation."""
    markdown_path = Path(path)
    chunks = split_markdown_file(
        markdown_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **splitter_kwargs,
    )

    print(f"File: {markdown_path.resolve()}")
    print(f"Total chunks: {len(chunks)}\n")

    for index, chunk in enumerate(chunks):
        header = chunk.metadata.get("header_2") or chunk.metadata.get("header_3") or ""
        chunk_type = chunk.metadata.get("chunk_type", "text")
        print("=" * 72)
        print(f"CHUNK {index}  type={chunk_type}")
        if header:
            print(f"Section: {header}")
        if chunk_type in {"code", "code_summary"}:
            print(
                "code_language={lang}  code_block_index={block}  code_ref={ref}".format(
                    lang=chunk.metadata.get("code_language"),
                    block=chunk.metadata.get("code_block_index"),
                    ref=chunk.metadata.get("code_ref"),
                )
            )
            if chunk.metadata.get("code_operations"):
                print(f"Operations: {', '.join(chunk.metadata['code_operations'])}")
            if chunk.metadata.get("code_entities"):
                entities = chunk.metadata["code_entities"]
                preview = ", ".join(entities[:8])
                if len(entities) > 8:
                    preview += f", ... (+{len(entities) - 8})"
                print(f"Entities: {preview}")
            if chunk.metadata.get("raw_code"):
                print(f"raw_code length: {len(chunk.metadata['raw_code'])}")
            if chunk.metadata.get("caption"):
                print(f"Caption: {chunk.metadata['caption']}")
        print(
            f"section_index={chunk.metadata.get('section_index')}  "
            f"chunk_index={chunk.metadata.get('chunk_index')}"
        )
        print("-" * 72)
        print(chunk.page_content)
        print()

    return chunks


def print_code_chunks_from_file(
    path: str | Path = DEFAULT_DEBEZIUM_MARKDOWN,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    **splitter_kwargs: Any,
) -> list[Document]:
    """Print only code-related chunks for manual validation."""
    markdown_path = Path(path)
    chunks = split_markdown_file(
        markdown_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **splitter_kwargs,
    )
    code_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("chunk_type") in {"code", "code_summary"}
    ]

    print(f"File: {markdown_path.resolve()}")
    print(f"Total chunks: {len(chunks)}  code-related chunks: {len(code_chunks)}\n")

    for index, chunk in enumerate(code_chunks):
        chunk_type = chunk.metadata.get("chunk_type", "code")
        print("=" * 72)
        print(f"CODE CHUNK {index}  type={chunk_type}")
        print(f"code_block_index={chunk.metadata.get('code_block_index')}")
        print(f"code_language={chunk.metadata.get('code_language')}")
        if chunk.metadata.get("code_ref"):
            print(f"code_ref={chunk.metadata['code_ref']}")
        if chunk.metadata.get("code_operations"):
            print(f"Operations: {', '.join(chunk.metadata['code_operations'])}")
        if chunk.metadata.get("code_entities"):
            entities = chunk.metadata["code_entities"]
            preview = ", ".join(entities[:8])
            if len(entities) > 8:
                preview += f", ... (+{len(entities) - 8})"
            print(f"Entities: {preview}")
        if chunk.metadata.get("raw_code"):
            print(f"raw_code length: {len(chunk.metadata['raw_code'])}")
        if chunk.metadata.get("local_heading"):
            print(f"Step: {chunk.metadata['local_heading']}")
        if chunk.metadata.get("caption"):
            print(f"Caption: {chunk.metadata['caption']}")
        header = chunk.metadata.get("header_3") or chunk.metadata.get("header_4") or ""
        if header:
            print(f"Section: {header}")
        print("-" * 72)
        print(chunk.page_content)
        print()

    return code_chunks


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--code":
        print_code_chunks_from_file(DEFAULT_DEBEZIUM_MARKDOWN)
    else:
        target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VALIDATION_MARKDOWN
        print_chunks_from_file(target)
