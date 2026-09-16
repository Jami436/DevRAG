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


class _MappingRegistry:
    """Registry stub that routes each extension to its own parser."""

    def __init__(self, parsers: dict[str, _StubParser]) -> None:
        self._parsers = parsers

    def get_parser(self, file_path: Path) -> _StubParser:
        return self._parsers[file_path.suffix.lower()]


class _FailingParser:
    def parse(self, file_path: Path) -> Document:
        raise FileNotFoundError(file_path)


class _FailingRegistry:
    def get_parser(self, file_path: Path) -> _FailingParser:
        return _FailingParser()


def _document(title: str = "Guide") -> Document:
    return Document(
        title=title,
        source="markdown",
        pages=[DocumentPage(page_number=1, content="Hello")],
    )


def test_execute_returns_document_from_selected_parser() -> None:
    expected = _document()
    parser = _StubParser(expected)
    use_case = IngestDocument(_MappingRegistry({".md": parser}))

    result = use_case.execute(Path("guide.md"))

    assert result is expected
    assert result.page_count == 1


def test_execute_passes_file_path_to_parser() -> None:
    parser = _StubParser(_document())
    use_case = IngestDocument(_MappingRegistry({".md": parser}))
    file_path = Path("docs", "guide.md")

    use_case.execute(file_path)

    assert parser.calls == [file_path]


def test_execute_selects_parser_by_file_extension() -> None:
    markdown_parser = _StubParser(_document("Markdown Guide"))
    html_parser = _StubParser(_document("HTML Guide"))
    use_case = IngestDocument(
        _MappingRegistry({".md": markdown_parser, ".html": html_parser})
    )

    markdown_result = use_case.execute(Path("guide.md"))
    html_result = use_case.execute(Path("guide.html"))

    assert markdown_result.title == "Markdown Guide"
    assert html_result.title == "HTML Guide"
    assert markdown_parser.calls == [Path("guide.md")]
    assert html_parser.calls == [Path("guide.html")]


def test_execute_propagates_parser_errors() -> None:
    use_case = IngestDocument(_FailingRegistry())

    with pytest.raises(FileNotFoundError):
        use_case.execute(Path("missing.pdf"))
