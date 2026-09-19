"""Integration tests for retrieval evaluation against live PostgreSQL.

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

from app.application.evaluation.evaluate_retrieval import EvaluateRetrieval
from app.domain.documents.chunks import DocumentChunk
from app.domain.documents.entities import Document, DocumentPage
from app.domain.evaluation.entities import GoldenQuery
from app.domain.retrieval.entities import RetrievedChunk
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


def test_evaluation_scores_hybrid_retrieval(
    pg_env: tuple[Engine, list[UUID]], session: Session
) -> None:
    _engine, written_ids = pg_env
    repo = DocumentRepository(session)
    document = Document(
        title="Vector RAG Guide",
        source="vector-rag.md",
        pages=[DocumentPage(page_number=1, content="content")],
    )
    chunks = [
        DocumentChunk(
            document_id=document.id,
            content="hybrid search fuses vector and full-text signals",
            chunk_index=0,
            page_number=1,
            metadata={"heading": "Retrieval"},
        ),
        DocumentChunk(
            document_id=document.id,
            content="reranking reorders chunks by cross-encoder scores",
            chunk_index=1,
            page_number=1,
            metadata={"heading": "Reranking"},
        ),
    ]
    repo.persist(
        document, chunks, embeddings=[_vector(1.0), _vector(0.5)]
    )
    session.commit()
    written_ids.append(document.id)

    golden = [
        GoldenQuery(
            query="how does hybrid search fuse signals?",
            relevant_chunk_ids=[chunks[0].id],
        ),
        GoldenQuery(
            query="what reranks the candidate chunks?",
            relevant_chunk_ids=[chunks[1].id],
        ),
    ]

    def retriever(query: str) -> list[RetrievedChunk]:
        return repo.hybrid_search(
            query_text=query,
            query_embedding=_vector(1.0),
            limit=5,
            candidates=10,
        )

    report = EvaluateRetrieval(
        loader=_ListLoader(golden),
        retriever=retriever,
        top_k=5,
    ).execute()

    assert report.num_queries == 2
    assert report.hit_rate == pytest.approx(1.0)
    assert report.mean_reciprocal_rank == pytest.approx(1.0)


class _ListLoader:
    """In-memory golden query loader used to keep the test self-contained."""

    def __init__(self, queries: list[GoldenQuery]) -> None:
        self._queries = queries

    def load(self) -> list[GoldenQuery]:
        return self._queries