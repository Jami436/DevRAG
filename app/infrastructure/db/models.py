from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.settings import settings
from app.infrastructure.db.base import Base

_VECTOR_DIMENSION = settings.embedding_dimension

_jsonb = JSONB().with_variant(JSON(), "sqlite")
embedding_type = Vector(_VECTOR_DIMENSION).with_variant(JSON(), "sqlite")


class DocumentModel(Base):
    """Persists a source document's descriptive metadata."""

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, str]] = mapped_column(
        _jsonb, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    chunks: Mapped[list["ChunkModel"]] = relationship(
        "ChunkModel",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChunkModel(Base):
    """Persists a document chunk along with its embedding vector."""

    __tablename__ = "chunks"
    __table_args__ = (
        # One chunk per (document, position) so re-ingesting is idempotent.
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_: Mapped[dict[str, str]] = mapped_column(
        _jsonb, nullable=False, default=dict
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        embedding_type,
        nullable=True,
    )

    document: Mapped[DocumentModel] = relationship(
        "DocumentModel", back_populates="chunks"
    )