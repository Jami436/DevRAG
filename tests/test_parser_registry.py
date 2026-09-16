from pathlib import Path

import pytest

from app.domain.documents.entities import Document, DocumentPage
from app.infrastructure.ingestion.parser_registry import (
    ParserRegistry,
    UnknownFileTypeError,
    build_default_parser_registry,
)


class _StubParser:
    """Parser stub that records the files it is asked to parse."""

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def parse(self, file_path: Path) -> Document:
        self.calls.append(file_path)
        return Document(
            title=file_path.stem,
            source="stub",
            pages=[DocumentPage(page_number=1, content="content")],
        )


def test_get_parser_returns_registered_parser() -> None:
    registry = ParserRegistry()
    parser = _StubParser()
    registry.register(parser, [".md"])

    assert registry.get_parser(Path("guide.md")) is parser


def test_get_parser_is_case_insensitive_on_extension() -> None:
    registry = ParserRegistry()
    parser = _StubParser()
    registry.register(parser, [".md"])

    assert registry.get_parser(Path("GUIDE.MD")) is parser
    assert registry.get_parser(Path("guide.Md")) is parser


def test_get_parser_accepts_extensions_without_dot() -> None:
    registry = ParserRegistry()
    parser = _StubParser()
    registry.register(parser, ["md"])

    assert registry.get_parser(Path("guide.md")) is parser


def test_get_parser_raises_for_unregistered_extension() -> None:
    registry = ParserRegistry()

    with pytest.raises(UnknownFileTypeError):
        registry.get_parser(Path("notes.txt"))


def test_parse_dispatches_to_registered_parser() -> None:
    registry = ParserRegistry()
    markdown_parser = _StubParser()
    html_parser = _StubParser()
    registry.register(markdown_parser, (".md", ".markdown"))
    registry.register(html_parser, (".html", ".htm"))

    markdown_document = registry.parse(Path("guide.md"))
    html_document = registry.parse(Path("guide.htm"))

    assert markdown_document.title == "guide"
    assert html_document.title == "guide"
    assert markdown_document.source == "stub"
    assert markdown_parser.calls == [Path("guide.md")]
    assert html_parser.calls == [Path("guide.htm")]


def test_register_overwrites_existing_extension_mapping() -> None:
    registry = ParserRegistry()
    first = _StubParser()
    second = _StubParser()
    registry.register(first, [".md"])
    registry.register(second, [".md"])

    assert registry.get_parser(Path("guide.md")) is second


def test_default_registry_parses_every_built_in_file_type(
    sample_pdf: Path,
    sample_markdown: Path,
    sample_html: Path,
) -> None:
    registry = build_default_parser_registry()

    pdf = registry.parse(sample_pdf)
    markdown = registry.parse(sample_markdown)
    html = registry.parse(sample_html)

    assert pdf.source == "pdf"
    assert markdown.source == "markdown"
    assert html.source == "html"
    assert pdf.page_count == 6
    assert markdown.page_count == 3
    assert html.page_count == 3


def test_default_registry_rejects_unknown_extension() -> None:
    registry = build_default_parser_registry()

    with pytest.raises(UnknownFileTypeError):
        registry.parse(Path("notes.txt"))
