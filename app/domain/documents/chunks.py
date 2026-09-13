from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(slots=True)
class DocumentChunk:
    """Represents a searchable chunk extracted from a document."""

    document_id: UUID
    content: str
    chunk_index: int
    page_number: int
    metadata: dict[str, str] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
