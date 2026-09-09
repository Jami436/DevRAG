from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4


@dataclass(slots=True)
class DocumentPage:
    """Represents a single page belonging to a document."""

    page_number: int
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    """Represents a source document composed of individual pages."""

    title: str
    source: str
    pages: list[DocumentPage]
    source_url: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def page_count(self) -> int:
        """Return the number of pages in the document."""
        return len(self.pages)