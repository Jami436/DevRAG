from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.documents.chunks import DocumentChunk
from app.domain.documents.entities import Document
from app.infrastructure.db.models import ChunkModel, DocumentModel


@dataclass(slots=True)
class ChunkSearchResult:
    """A retrieved chunk paired with its cosine distance score."""

    chunk_id: UUID
    document_id: UUID
    content: str
    chunk_index: int
    page_number: int
    metadata: dict[str, str]
    score: float


class DocumentRepository:
    """Persist documents and chunks and run vector similarity queries."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_document(self, document: Document) -> None:
        model = self._session.get(DocumentModel, document.id)
        if model is None:
            model = DocumentModel(
                id=document.id,
                title=document.title,
                source=document.source,
                source_url=document.source_url,
                metadata_=dict(document.metadata),
                created_at=document.created_at,
            )
            self._session.add(model)
        else:
            model.title = document.title
            model.source = document.source
            model.source_url = document.source_url
            model.metadata_ = dict(document.metadata)
        self._session.flush()

    def persist(
        self,
        document: Document,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Persist a document and all of its chunk embeddings atomically.

        The write is idempotent and self-consistent: re-ingesting the same
        document replaces its chunk rows in place and prunes any stored chunk
        whose index is no longer part of the ingest, so the stored state always
        mirrors the last call exactly. All changes share one transaction; a
        caller that rolls back on failure undoes the document record too.
        """
        self.save_document(document)
        self.upsert_chunks(chunks, embeddings)
        expected_indices = {chunk.chunk_index for chunk in chunks}
        for chunk in self.get_chunks(document.id):
            if chunk.chunk_index not in expected_indices:
                self._session.delete(chunk)
        self._session.flush()

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Insert or update chunks keyed by ``(document_id, chunk_index)``.

        A single-pass UPSERT keeps re-ingestion idempotent: existing rows are
        refreshed in place and new rows are inserted without orphaning old data.
        """
        if embeddings is None:
            vectorized: Sequence[list[float] | None] = [None] * len(chunks)
        else:
            if len(embeddings) != len(chunks):
                raise ValueError("embeddings must align one-to-one with chunks")
            vectorized = embeddings
        for chunk, vector in zip(chunks, vectorized, strict=True):
            existing = self._session.scalar(
                select(ChunkModel).where(
                    ChunkModel.document_id == chunk.document_id,
                    ChunkModel.chunk_index == chunk.chunk_index,
                )
            )
            if existing is None:
                self._session.add(
                    ChunkModel(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        chunk_index=chunk.chunk_index,
                        page_number=chunk.page_number,
                        metadata_=dict(chunk.metadata),
                        embedding=vector,
                    )
                )
            else:
                existing.content = chunk.content
                existing.page_number = chunk.page_number
                existing.metadata_ = dict(chunk.metadata)
                existing.embedding = vector
        self._session.flush()

    def get_document(self, document_id: UUID) -> DocumentModel | None:
        return self._session.get(DocumentModel, document_id)

    def get_document_by_source(self, source: str) -> DocumentModel | None:
        return self._session.scalar(
            select(DocumentModel).where(DocumentModel.source == source)
        )

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[DocumentModel]:
        return list(
            self._session.scalars(
                select(DocumentModel)
                .order_by(DocumentModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        )

    def count_documents(self) -> int:
        return self._session.scalar(select(func.count(DocumentModel.id))) or 0

    def get_chunks(self, document_id: UUID) -> list[ChunkModel]:
        return list(
            self._session.scalars(
                select(ChunkModel)
                .where(ChunkModel.document_id == document_id)
                .order_by(ChunkModel.chunk_index)
            )
        )

    def delete_document(self, document_id: UUID) -> bool:
        model = self._session.get(DocumentModel, document_id)
        if model is None:
            return False
        self._session.delete(model)
        self._session.flush()
        return True

    def commit(self) -> None:
        self._session.commit()

    def close(self) -> None:
        self._session.close()

    def cosine_search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        distance_threshold: float | None = None,
    ) -> list[ChunkSearchResult]:
        """Return the chunks nearest to ``query_embedding`` by cosine distance.

        Only valid on a pgvector-backed database; the underlying operator
        (``<=>``) is PostgreSQL-specific.
        """
        distance = ChunkModel.embedding.cosine_distance(query_embedding)
        statement = select(ChunkModel, distance.label("distance"))
        if distance_threshold is not None:
            statement = statement.where(distance < distance_threshold)
        rows = self._session.execute(
            statement.order_by(distance).limit(limit)
        ).all()
        return [
            ChunkSearchResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                metadata=dict(chunk.metadata_),
                score=score,
            )
            for chunk, score in rows
        ]