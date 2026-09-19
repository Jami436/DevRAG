from typing import Protocol

from app.domain.retrieval.entities import RetrievedChunk


class Reranker(Protocol):
    """Re-order retrieved chunks by their relevance to the query."""

    def rerank(
        self, query: str, results: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Return ``results`` re-ordered by relevance to ``query``.

        Implementations update each chunk's ``final_score`` and ``rank``.
        """