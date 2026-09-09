from pathlib import Path
from typing import Protocol

from app.domain.documents.entities import Document

class DocumentParser(Protocol):
    def parse(self, file_path: Path) -> Document:
        """Parse a document from the given file path and return a Document object."""