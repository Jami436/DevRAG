from pathlib import Path

import pytest

from app.application.ingestion.ingest_document import IngestDocument
from app.domain.documents.entities import Document, DocumentPage


class _StubParser:
    """Minimal parser stub that returns a pre-built document."""

    def __init__(self, document: Document) -> None:
        self._document = document
        self.calls: list[Path] = []

    def parse(self, file_path: Path) -> Document:
        self.calls.append(file_path)
        return self._document


class _FailingParser:
    def parse(self, file_path: Path) -> Document:
        raise FileNotFoundError(file_path)


def test_execute_returns_document_from_parser() -> None:
    document = Document(
        title="Guide",
        source="markdown",
        pages=[DocumentPage(page_number=1, content="Hello")],
    )
    parser = _StubParser(document)
    use_case = IngestDocument(parser)

    result = use_case.execute(Path("guide.md"))

    assert result is document
    assert result.page_count == 1


def test_execute_passes_file_path_to_parser() -> None:
    parser = _StubParser(Document(title="Guide", source="markdown", pages=[]))
    use_case = IngestDocument(parser)
    file_path = Path("docs", "guide.md")

    use_case.execute(file_path)

    assert parser.calls == [file_path]


def test_execute_propagates_parser_errors() -> None:
    use_case = IngestDocument(_FailingParser())
    missing = Path("missing.pdf")

    with pytest.raises(FileNotFoundError):
        use_case.execute(missing)
