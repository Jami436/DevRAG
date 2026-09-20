from dataclasses import dataclass, field
from uuid import UUID


@dataclass(slots=True)
class Citation:
    """An answer passage tied back to the chunk it was sourced from.

    ``excerpt`` mirrors the truncated passage the language model was shown while
    generating, so consumers can display exactly what grounded the answer.
    """

    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    page_number: int
    document_title: str | None = None
    excerpt: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class GeneratedAnswer:
    """A grounded answer together with the sources it cites."""

    query: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    model: str | None = None