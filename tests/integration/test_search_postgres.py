"""Integration tests for hybrid search against live PostgreSQL.

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
from app.infrastructure.db.repositories.document_repository import (
    DocumentRepository,
)

_DATABASE_URL = os.environ.get("DATABASE_URL", "")

requires_live_postgres = pytest.mark.skipif(
    not _DATABASE_URL.startswith("postgresql"),
    reason="DATABASE_URL must point to a live PostgreSQL with pgvector",
)

pytestmark = [requires_live_postgres, pytest.mark.integration]


def _vector(value: float = 1.0) -> list[float]:
    """Build a 1536-dimensioned vector matching the schema's vector column."""
    return [value] + [0.0] * 1535


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
            delete(DocumentModel).where(DocumentModel.id.in_(written_ids))
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


@pytest.fixture()
def seed_data(
    pg_env: tuple[Engine, list[UUID]], session: Session
) -> dict[int, UUID]:
    """Persist chunks with distinct vector and keyword signals.

    * index 0: vector-equal to the query and matches the keyword ``search``.
    * index 1: vector-close to the query but keyword-miss.
    * index 2: vector-far from the query and keyword-miss.
    """
    _engine, written_ids = pg_env
    repo = DocumentRepository(session)
    document = Document(
        title="Hybrid Sources",
        source="hybrid.md",
        pages=[DocumentPage(page_number=1, content="content")],
    )
    chunks = [
        DocumentChunk(
            document_id=document.id,
            content="vector database search with embeddings and ranking",
            chunk_index=0,
            page_number=1,
            metadata={"heading": "Retrieval"},
        ),
        DocumentChunk(
            document_id=document.id,
            content="postgresql connection pooling is configured here",
            chunk_index=1,
            page_number=1,
            metadata={"heading": "Deployment"},
        ),
        DocumentChunk(
            document_id=document.id,
            content="cooking pasta recipes with fresh tomatoes",
            chunk_index=2,
            page_number=1,
            metadata={"heading": "Recipes"},
        ),
    ]
    repo.persist(
        document,
        chunks,
        embeddings=[_vector(1.0), _vector(0.9), _vector(-1.0)],
    )
    session.commit()
    written_ids.append(document.id)
    return {chunk.chunk_index: chunk.id for chunk in chunks}


def test_keyword_search_ranks_matching_chunks_first(
    session: Session, seed_data: dict[int, UUID]
) -> None:
    repo = DocumentRepository(session)

    results = repo.keyword_search("search", limit=5)

    assert len(results) == 1
    assert results[0].chunk_id == seed_data[0]
    assert results[0].document_title == "Hybrid Sources"
    assert results[0].keyword_score is not None
    assert results[0].keyword_score > 0.0


def test_keyword_search_returns_empty_for_no_matches(
    session: Session, seed_data: dict[int, UUID]
) -> None:
    repo = DocumentRepository(session)

    results = repo.keyword_search("nonexistenttermxyz", limit=5)

    assert results == []


def test_hybrid_search_fuses_vector_and_keyword_signals(
    session: Session, seed_data: dict[int, UUID]
) -> None:
    repo = DocumentRepository(session)

    results = repo.hybrid_search(
        query_text="search",
        query_embedding=_vector(1.0),
        limit=3,
        candidates=10,
    )

    assert [result.chunk_index for result in results] == [0, 1, 2]
    assert results[0].chunk_id == seed_data[0]
    assert results[0].rank == 1
    assert results[0].vector_score == pytest.approx(1.0)
    assert results[0].keyword_score is not None
    assert results[0].keyword_score > 0.0
    assert results[0].document_title == "Hybrid Sources"
    fused_scores = [result.final_score or 0.0 for result in results]
    assert fused_scores[0] > fused_scores[1] > fused_scores[2]


def test_hybrid_search_truncates_to_limit(
    session: Session, seed_data: dict[int, UUID]
) -> None:
    repo = DocumentRepository(session)

    results = repo.hybrid_search(
        query_text="search",
        query_embedding=_vector(1.0),
        limit=1,
        candidates=10,
    )

    assert len(results) == 1
    assert results[0].chunk_index == 0