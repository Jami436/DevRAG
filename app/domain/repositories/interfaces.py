from typing import Protocol

from app.domain.documents.chunks import DocumentChunk
from app.domain.documents.entities import Document
from app.domain.retrieval.entities import RetrievedChunk


class DocumentRepository(Protocol):
    """Write contract for persisting documents, chunks and their embeddings.

    Concrete implementations may expose additional read and search methods with
    storage-specific return types; this protocol pins down only the operations an
    application use case needs, expressed purely in domain terms.
    """

    def save_document(self, document: Document) -> None:
        """Insert or update the document's descriptive record."""

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Insert or update chunks keyed by ``(document_id, chunk_index)``."""

    def persist(
        self,
        document: Document,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Persist a document together with all of its chunk embeddings."""

    def commit(self) -> None:
        """Commit the current transaction."""

    def close(self) -> None:
        """Release the underlying session or resources."""


class SearchRepository(Protocol):
    """Read contract for multi-signal retrieval over persisted chunks."""

    def keyword_search(
        self, query_text: str, *, limit: int = 10
    ) -> list[RetrievedChunk]:
        """Run a full-text keyword search over chunk contents."""

    def hybrid_search(
        self,
        *,
        query_text: str,
        query_embedding: list[float],
        limit: int = 10,
        candidates: int = 50,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        fusion_k: int = 60,
        distance_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Fuse vector and keyword results into a single ranked list."""

    def close(self) -> None:
        """Release the underlying session or resources."""