from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.documents.chunks import DocumentChunk
from app.domain.documents.entities import Document
from app.domain.retrieval.entities import RetrievedChunk
from app.domain.retrieval.fusion import reciprocal_rank_fusion
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
        hits = self._cosine_hits(query_embedding, limit, distance_threshold)
        results: list[ChunkSearchResult] = []
        for hit in hits:
            vector_score = hit.vector_score
            assert vector_score is not None
            results.append(
                ChunkSearchResult(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    content=hit.content,
                    chunk_index=hit.chunk_index,
                    page_number=hit.page_number,
                    metadata=hit.metadata,
                    score=1.0 - vector_score,
                )
            )
        return results

    def keyword_search(
        self, query_text: str, *, limit: int = 10
    ) -> list[RetrievedChunk]:
        """Search chunk contents with PostgreSQL full-text search.

        Matching uses the ``english`` text search configuration and results are
        ranked by ``ts_rank_cd``. The tsvector is derived inline from
        ``content`` so the query also benefits from the stored generated column
        added by migration ``002`` when it is present. Only valid on a
        PostgreSQL-backed database.
        """
        tsvector = func.to_tsvector("english", ChunkModel.content)
        tsquery = func.websearch_to_tsquery("english", query_text)
        rank = func.ts_rank_cd(tsvector, tsquery).label("rank")
        statement = (
            select(ChunkModel, DocumentModel.title, rank)
            .join(DocumentModel, ChunkModel.document_id == DocumentModel.id)
            .where(tsvector.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(limit)
        )
        rows = self._session.execute(statement).all()
        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_title=title,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                metadata=dict(chunk.metadata_),
                keyword_score=float(score),
            )
            for chunk, title, score in rows
        ]

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
        """Fuse dense vector and full-text results with Reciprocal Rank Fusion.

        Each signal is fetched with ``candidates`` hits (typically larger than
        ``limit`` so the fusion has enough spanning candidates) and the merged
        list is truncated to ``limit``.
        """
        vector_hits = self._cosine_hits(
            query_embedding, candidates, distance_threshold
        )
        keyword_hits = self.keyword_search(query_text, limit=candidates)
        return reciprocal_rank_fusion(
            (vector_hits, keyword_hits),
            weights=(vector_weight, keyword_weight),
            k=fusion_k,
            limit=limit,
        )

    def _cosine_hits(
        self,
        query_embedding: list[float],
        limit: int,
        distance_threshold: float | None,
    ) -> list[RetrievedChunk]:
        """Run the vector similarity query mapped to domain search results.

        The stored cosine ``distance`` is converted to a similarity score
        (``1 - distance``) so that higher always means more relevant.
        """
        distance = ChunkModel.embedding.cosine_distance(query_embedding)
        statement = select(ChunkModel, distance.label("distance"))
        if distance_threshold is not None:
            statement = statement.where(distance < distance_threshold)
        rows = self._session.execute(
            statement.order_by(distance).limit(limit)
        ).all()
        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                metadata=dict(chunk.metadata_),
                vector_score=1.0 - float(distance),
            )
            for chunk, distance in rows
        ]