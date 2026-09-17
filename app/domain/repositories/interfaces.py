from typing import Protocol

from app.domain.documents.chunks import DocumentChunk
from app.domain.documents.entities import Document


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