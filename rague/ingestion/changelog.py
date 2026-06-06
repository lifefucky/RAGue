"""Changelog reporting for ingestion runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document


CODE_CHUNK_TYPES = frozenset({"code", "code_summary"})


@dataclass
class ChunkRunStats:
    """Aggregate chunk statistics for one ingestion run."""

    total: int = 0
    chars_total: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    per_page_counts: list[int] = field(default_factory=list)

    def record_page(self, chunks: list[Document]) -> None:
        if not chunks:
            return

        page_count = len(chunks)
        self.total += page_count
        self.per_page_counts.append(page_count)
        for chunk in chunks:
            chunk_type = chunk.metadata.get("chunk_type", "text")
            self.by_type[chunk_type] = self.by_type.get(chunk_type, 0) + 1
            self.chars_total += len(chunk.page_content)

    @property
    def avg_per_page(self) -> float:
        if not self.per_page_counts:
            return 0.0
        return self.total / len(self.per_page_counts)

    @property
    def min_per_page(self) -> int | None:
        if not self.per_page_counts:
            return None
        return min(self.per_page_counts)

    @property
    def max_per_page(self) -> int | None:
        if not self.per_page_counts:
            return None
        return max(self.per_page_counts)

    def format_by_type(self) -> str:
        if not self.by_type:
            return "none"
        return ", ".join(
            f"{chunk_type}={count}"
            for chunk_type, count in sorted(self.by_type.items())
        )

    @property
    def code_count(self) -> int:
        return self.by_type.get("code", 0)

    @property
    def code_summary_count(self) -> int:
        return self.by_type.get("code_summary", 0)

    @property
    def code_counts_by_type(self) -> dict[str, int]:
        return {
            chunk_type: self.by_type.get(chunk_type, 0)
            for chunk_type in sorted(CODE_CHUNK_TYPES)
        }

    @property
    def code_fragments_total(self) -> int:
        return sum(self.by_type.get(chunk_type, 0) for chunk_type in CODE_CHUNK_TYPES)

    def format_code_fragments(self) -> str:
        total = self.code_fragments_total
        if total == 0:
            return "total=0"
        return (
            f"total={total} "
            f"(code={self.code_count}, code_summary={self.code_summary_count})"
        )


@dataclass
class IngestionRunReport:
    """Collect statistics and notes for one ingestion run."""

    scope: str
    collection_name: str
    embedding_provider: str
    embedding_model: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    pages_discovered: int = 0
    pages_loaded: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    chunks_created: int = 0
    points_deleted: int = 0
    points_upserted: int = 0
    attachments_discovered: int = 0
    attachment_samples_saved: int = 0
    attachments_skipped: int = 0
    attachments_failed: int = 0
    attachment_sample_extensions: list[str] = field(default_factory=list)
    chunk_stats: ChunkRunStats = field(default_factory=ChunkRunStats)
    worked: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    config_summary: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    changelog_path: Path | None = None

    def mark_finished(self) -> None:
        self.finished_at = datetime.now(timezone.utc)

    def format_attachment_extensions(self) -> str:
        return ", ".join(sorted(set(self.attachment_sample_extensions))) or "none"

    def format_attachment_summary(self) -> str:
        extensions = self.format_attachment_extensions()
        return (
            f"discovered={self.attachments_discovered}, "
            f"samples_saved={self.attachment_samples_saved}, "
            f"skipped={self.attachments_skipped}, "
            f"failed={self.attachments_failed}, "
            f"extensions={extensions}"
        )

    def write_markdown(self, output_dir: str | Path) -> Path:
        self.mark_finished()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = self.started_at.strftime("%Y-%m-%d_%H-%M-%S")
        file_path = output_path / f"{timestamp}_confluence_ingestion.md"
        file_path.write_text(self._render_markdown(), encoding="utf-8")
        self.changelog_path = file_path
        return file_path

    def print_summary(self) -> None:
        """Print a final run summary (always visible in the terminal)."""
        duration = (
            f"{self.duration_seconds:.1f}s"
            if self.duration_seconds is not None
            else "n/a"
        )
        stats = self.chunk_stats
        min_per_page = stats.min_per_page if stats.min_per_page is not None else 0
        max_per_page = stats.max_per_page if stats.max_per_page is not None else 0

        lines = [
            f"Ingestion finished in {duration}",
            (
                "Pages: "
                f"discovered={self.pages_discovered} "
                f"loaded={self.pages_loaded} "
                f"skipped={self.pages_skipped} "
                f"failed={self.pages_failed}"
            ),
            (
                "Chunks: "
                f"total={stats.total} | "
                f"per page avg={stats.avg_per_page:.1f} "
                f"min={min_per_page} max={max_per_page}"
            ),
            f"Chunks by type: {stats.format_by_type()}",
            f"Code fragments: {stats.format_code_fragments()}",
            f"Attachments: {self.format_attachment_summary()}",
            (
                "Qdrant: "
                f"deleted={self.points_deleted} upserted={self.points_upserted}"
            ),
        ]
        if self.changelog_path is not None:
            lines.append(f"Changelog written: {self.changelog_path}")

        for line in lines:
            print(line)

    def _render_markdown(self) -> str:
        finished = self.finished_at.isoformat() if self.finished_at else "n/a"
        duration = (
            f"{self.duration_seconds:.1f}s"
            if self.duration_seconds is not None
            else "n/a"
        )
        stats = self.chunk_stats
        lines = [
            "# Confluence Ingestion Changelog",
            "",
            "## Run Summary",
            "",
            f"- Started at: `{self.started_at.isoformat()}`",
            f"- Finished at: `{finished}`",
            f"- Duration: `{duration}`",
            f"- Scope: `{self.scope}`",
            f"- Qdrant collection: `{self.collection_name}`",
            f"- Embedding provider: `{self.embedding_provider}`",
            f"- Embedding model: `{self.embedding_model}`",
            "",
            "## Config",
            "",
        ]

        if self.config_summary:
            for key, value in self.config_summary.items():
                lines.append(f"- `{key}`: `{value}`")
        else:
            lines.append("- No additional config recorded.")

        lines.extend(
            [
                "",
                "## Metrics",
                "",
                f"- Pages discovered: `{self.pages_discovered}`",
                f"- Pages loaded: `{self.pages_loaded}`",
                f"- Pages skipped: `{self.pages_skipped}`",
                f"- Pages failed: `{self.pages_failed}`",
                f"- Chunks created: `{self.chunks_created}`",
                f"- Points deleted: `{self.points_deleted}`",
                f"- Points upserted: `{self.points_upserted}`",
                f"- Attachments discovered: `{self.attachments_discovered}`",
                f"- Attachment samples saved: `{self.attachment_samples_saved}`",
                f"- Attachments skipped: `{self.attachments_skipped}`",
                f"- Attachments failed: `{self.attachments_failed}`",
                "- Attachment sample extensions: "
                f"`{self.format_attachment_extensions()}`",
                "",
                "## Chunk Summary",
                "",
                f"- Total chunks: `{stats.total}`",
            ]
        )

        if stats.per_page_counts:
            lines.extend(
                [
                    (
                        "- Chunks per page: "
                        f"avg `{stats.avg_per_page:.1f}`, "
                        f"min `{stats.min_per_page}`, "
                        f"max `{stats.max_per_page}`"
                    ),
                    f"- Total chunk characters: `{stats.chars_total}`",
                    "- By type:",
                ]
            )
            for chunk_type, count in sorted(stats.by_type.items()):
                lines.append(f"  - `{chunk_type}`: `{count}`")
            lines.append(
                (
                    "- Code fragments: "
                    f"`{stats.code_fragments_total}` "
                    f"(`code`: `{stats.code_count}`, "
                    f"`code_summary`: `{stats.code_summary_count}`)"
                )
            )
        else:
            lines.append("- No chunks recorded.")
            lines.append(
                (
                    "- Code fragments: `0` "
                    "(`code`: `0`, `code_summary`: `0`)"
                )
            )

        lines.extend(
            [
                "",
                "## Attachments",
                "",
                f"- Discovered: `{self.attachments_discovered}`",
                f"- Samples saved: `{self.attachment_samples_saved}`",
                f"- Skipped (duplicate extension): `{self.attachments_skipped}`",
                f"- Failed: `{self.attachments_failed}`",
                (
                    "- Sample extensions: "
                    f"`{self.format_attachment_extensions()}`"
                ),
            ]
        )

        lines.extend(["", "## What Worked", ""])
        lines.extend(f"- {item}" for item in self.worked or ["Nothing recorded."])

        lines.extend(["", "## What Did Not Work", ""])
        lines.extend(f"- {item}" for item in self.failed or ["Nothing recorded."])

        lines.extend(["", "## Added", ""])
        lines.extend(f"- {item}" for item in self.added or ["Nothing recorded."])

        lines.extend(["", "## Removed Or Skipped", ""])
        lines.extend(f"- {item}" for item in self.removed or ["Nothing recorded."])

        if self.errors:
            lines.extend(["", "## Errors", ""])
            lines.extend(f"- {error}" for error in self.errors)

        lines.append("")
        return "\n".join(lines)
