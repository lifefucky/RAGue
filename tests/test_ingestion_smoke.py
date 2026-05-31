from pathlib import Path

from rague.sources.confluence.multi_page_loader import ConfluenceMultiPageLoader


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def get(self, url: str) -> FakeResponse:
        return FakeResponse(f"downloaded from {url}".encode("utf-8"))


class FakeConfluence:
    session = FakeSession()

    def get_page_by_id(self, page_id: str, expand: str = "") -> dict:
        return {
            "id": str(page_id),
            "title": "Page",
            "version": {"number": 1, "when": "2026-05-25T12:00:00.000Z"},
            "space": {"key": "DEV"},
            "ancestors": [],
            "body": {"view": {"value": "<p>body</p>"}},
        }

    def get_attachments_from_content(
        self,
        page_id: str,
        start: int = 0,
        limit: int = 100,
        expand: str = "",
    ) -> dict:
        if start:
            return {"results": []}
        return {
            "results": [
                {
                    "id": "a1",
                    "title": "first.pdf",
                    "_links": {"download": "/download/attachments/1/first.pdf"},
                },
                {
                    "id": "a2",
                    "title": "second.pdf",
                    "_links": {"download": "/download/attachments/1/second.pdf"},
                },
                {
                    "id": "a3",
                    "title": "slides.pptx",
                    "_links": {"download": "/download/attachments/1/slides.pptx"},
                },
            ]
        }


def test_attachment_samples_save_one_file_per_extension(tmp_path: Path) -> None:
    loader = ConfluenceMultiPageLoader(
        url="https://wiki.example",
        page_ids=["1"],
        confluence=FakeConfluence(),
    )

    seen_extensions: set[str] = set()
    stats = loader.save_attachment_samples("1", tmp_path, seen_extensions)

    assert stats["discovered"] == 3
    assert stats["saved"] == 2
    assert stats["skipped"] == 1
    assert stats["failed"] == 0
    assert seen_extensions == {"pdf", "pptx"}
    assert len(list(tmp_path.iterdir())) == 2
