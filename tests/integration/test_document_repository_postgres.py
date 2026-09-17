"""Integration tests for the document repository against live PostgreSQL.

These tests require PostgreSQL with the pgvector extension and opt in via the
``DATABASE_URL`` environment variable pointing at a ``postgresql`` connection
string. They expect a disposable database: the schema is created on demand and
any document rows written by the tests are removed afterwards.
"""

import os
from collections.abc import Iterator
from uuid import UUID

import pytest
from sqlalchemy import create_engine, delete, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.domain.documents.chunks import DocumentChunk
from app.domain.documents.entities import Document, DocumentPage
from app.infrastructure.db.base import Base
from app.infrastructure.db.models import DocumentModel
from app.infrastructure.db.repositories.document_repository import DocumentRepository

_DATABASE_URL = os.environ.get("DATABASE_URL", "")

requires_live_postgres = pytest.mark.skipif(
    not _DATABASE_URL.startswith("postgresql"),
    reason="DATABASE_URL must point to a live PostgreSQL with pgvector",
)

pytestmark = [requires_live_postgres, pytest.mark.integration]


def _vector(only_dimension_zero: float = 1.0) -> list[float]:
    """Build a 1536-dimensioned vector matching the schema's vector column."""
    return [only_dimension_zero] + [0.0] * 1535


@pytest.fixture(scope="module")
def pg_env() -> Iterator[tuple[Engine, list[UUID]]]:
    engine = create_engine(_DATABASE_URL, poolclass=NullPool)
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    pre_existing = inspect(engine).has_table(DocumentModel.__tablename__)
    if not pre_existing:
        Base.metadata.create_all(engine)

    written_ids: list[UUID] = []
    yield engine, written_ids

    with Session(engine) as cleanup:
        cleanup.execute(
            delete(DocumentModel).where(
                DocumentModel.id.in_(written_ids)
            )
        )
        cleanup.commit()
    if not pre_existing:
        Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def session(pg_env: tuple[Engine, list[UUID]]) -> Iterator[Session]:
    engine, _written_ids = pg_env
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        yield session


def _make_document(title: str, source: str) -> Document:
    return Document(
        title=title,
        source=source,
        pages=[DocumentPage(page_number=1, content="content")],
    )


def _make_chunk(
    document_id: UUID,
    index: int,
    content: str,
    heading: str,
) -> DocumentChunk:
    return DocumentChunk(
        document_id=document_id,
        content=content,
        chunk_index=index,
        page_number=index + 1,
        metadata={"heading": heading},
    )


def test_persist_round_trips_on_postgres(
    pg_env: tuple[Engine, list[UUID]], session: Session
) -> None:
    engine, written_ids = pg_env
    repo = DocumentRepository(session)
    document = _make_document("Postgres Guide", "postgres.md")
    chunks = [
        _make_chunk(document.id, 0, "chunk zero", "Intro"),
        _make_chunk(document.id, 1, "chunk one", "Details"),
    ]

    repo.persist(document, chunks, embeddings=[_vector(), _vector(0.5)])
    session.commit()
    written_ids.append(document.id)

    stored = repo.get_chunks(document.id)
    assert [chunk.content for chunk in stored] == ["chunk zero", "chunk one"]
    assert stored[0].embedding is not None
    assert len(stored[0].embedding) == 1536
    assert engine.dialect.name == "postgresql"


def test_cosine_search_orders_nearest_chunks_first(
    pg_env: tuple[Engine, list[UUID]], session: Session
) -> None:
    _, written_ids = pg_env
    repo = DocumentRepository(session)
    document = _make_document("Ordering", "ordering.md")
    query = _vector()  # ones in dimension zero, zeros elsewhere
    chunks = [
        _make_chunk(document.id, 0, "identical vector", "A"),
        _make_chunk(document.id, 1, "opposite vector", "B"),
    ]

    repo.persist(
        document,
        chunks,
        embeddings=[_vector(), _vector(-1.0)],
    )
    session.commit()
    written_ids.append(document.id)

    results = repo.cosine_search(query_embedding=query, limit=2)

    assert [result.chunk_index for result in results] == [0, 1]
    assert results[0].score == pytest.approx(0.0)
    assert results[0].score < results[1].score


def test_cosine_search_respects_distance_threshold(
    pg_env: tuple[Engine, list[UUID]], session: Session
) -> None:
    _, written_ids = pg_env
    repo = DocumentRepository(session)
    document = _make_document("Threshold", "threshold.md")
    query = _vector()
    chunks = [
        _make_chunk(document.id, 0, "identical vector", "A"),
        _make_chunk(document.id, 1, "opposite vector", "B"),
    ]

    repo.persist(
        document,
        chunks,
        embeddings=[_vector(), _vector(-1.0)],
    )
    session.commit()
    written_ids.append(document.id)

    results = repo.cosine_search(
        query_embedding=query, limit=10, distance_threshold=0.5
    )

    assert [result.chunk_index for result in results] == [0]
    assert results[0].score == pytest.approx(0.0)