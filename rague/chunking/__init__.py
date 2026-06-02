"""Chunking utilities for normalized source documents."""

__all__ = [
    "DEFAULT_HEADERS_TO_SPLIT_ON",
    "MarkdownDocumentTextSplitter",
]


def __getattr__(name: str):
    if name in __all__:
        from rague.chunking.markdown import (
            DEFAULT_HEADERS_TO_SPLIT_ON,
            MarkdownDocumentTextSplitter,
        )

        exports = {
            "DEFAULT_HEADERS_TO_SPLIT_ON": DEFAULT_HEADERS_TO_SPLIT_ON,
            "MarkdownDocumentTextSplitter": MarkdownDocumentTextSplitter,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
