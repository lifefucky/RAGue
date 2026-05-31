"""Changelog reporting for ingestion runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class IngestionRunReport:
    """Collect statistics and notes for one ingestion run."""

    scope: str
    collection_name: str
    embedding_provider: str
    embedding_model: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
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
    worked: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    config_summary: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def mark_finished(self) -> None:
        self.finished_at = datetime.now(timezone.utc)

    def write_markdown(self, output_dir: str | Path) -> Path:
        self.mark_finished()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = self.started_at.strftime("%Y-%m-%d_%H-%M-%S")
        file_path = output_path / f"{timestamp}_confluence_ingestion.md"
        file_path.write_text(self._render_markdown(), encoding="utf-8")
        return file_path

    def _render_markdown(self) -> str:
        finished = self.finished_at.isoformat() if self.finished_at else "n/a"
        lines = [
            "# Confluence Ingestion Changelog",
            "",
            "## Run Summary",
            "",
            f"- Started at: `{self.started_at.isoformat()}`",
            f"- Finished at: `{finished}`",
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
                f"`{', '.join(self.attachment_sample_extensions) or 'none'}`",
                "",
                "## What Worked",
                "",
            ]
        )
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
