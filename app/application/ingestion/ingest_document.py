from pathlib import Path

from app.domain.documents.entities import Document
from app.domain.documents.interfaces import DocumentParser


class IngestDocument:
    def __init__(self, parser: DocumentParser) -> None:
        self._parser = parser

    def execute(self, file_path: Path) -> Document:
        return self._parser.parse(file_path)
