from dataclasses import dataclass, field
from uuid import UUID


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk surfaced by a search signal, with per-signal and fused scores.

    ``vector_score`` and ``keyword_score`` are higher-is-better (cosine
    similarity and full-text rank respectively); ``final_score`` carries the
    fused or reranked score and ``rank`` the 1-based position after the last
    reordering step.
    """

    chunk_id: UUID
    document_id: UUID
    content: str
    chunk_index: int
    page_number: int
    metadata: dict[str, str] = field(default_factory=dict)
    document_title: str | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    final_score: float | None = None
    rank: int | None = None