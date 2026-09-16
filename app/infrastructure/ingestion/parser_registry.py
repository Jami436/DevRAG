from collections.abc import Iterable
from pathlib import Path

from app.domain.documents.entities import Document
from app.domain.documents.interfaces import DocumentParser


class UnknownFileTypeError(ValueError):
    """Raised when no parser is registered for the given file type."""


class ParserRegistry:
    """Dispatch document parsing based on file extension."""

    def __init__(self) -> None:
        self._parsers: dict[str, DocumentParser] = {}

    def register(self, parser: DocumentParser, extensions: Iterable[str]) -> None:
        for extension in extensions:
            self._parsers[self._normalize(extension)] = parser

    def get_parser(self, file_path: Path) -> DocumentParser:
        parser = self._parsers.get(self._normalize(file_path.suffix))
        if parser is None:
            raise UnknownFileTypeError(f"Unsupported file type: {file_path.suffix}")
        return parser

    def parse(self, file_path: Path) -> Document:
        return self.get_parser(file_path).parse(file_path)

    @staticmethod
    def _normalize(extension: str) -> str:
        normalized = extension.strip().lower()
        if normalized and not normalized.startswith("."):
            normalized = f".{normalized}"
        return normalized


def build_default_parser_registry() -> ParserRegistry:
    """Create a registry pre-loaded with every built-in parser."""
    from app.infrastructure.ingestion.html_parser import HTMLParser
    from app.infrastructure.ingestion.markdown_parser import MarkdownParser
    from app.infrastructure.ingestion.pdf_parser import PDFParser

    registry = ParserRegistry()
    registry.register(PDFParser(), PDFParser.SUPPORTED_EXTENSIONS)
    registry.register(MarkdownParser(), MarkdownParser.SUPPORTED_EXTENSIONS)
    registry.register(HTMLParser(), HTMLParser.SUPPORTED_EXTENSIONS)
    return registry
