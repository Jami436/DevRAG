from pathlib import Path

from app.domain.documents.entities import Document
from app.domain.documents.interfaces import ParserRegistry


class IngestDocument:
    """Ingest a document, selecting the parser by its file extension."""

    def __init__(self, parser_registry: ParserRegistry) -> None:
        self._parser_registry = parser_registry

    def execute(self, file_path: Path) -> Document:
        parser = self._parser_registry.get_parser(file_path)
        return parser.parse(file_path)
