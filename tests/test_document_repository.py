from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.documents.chunks import DocumentChunk
from app.domain.documents.entities import Document, DocumentPage
from app.domain.repositories.interfaces import (
    DocumentRepository as DocumentRepositoryProtocol,
)
from app.infrastructure.db.base import Base
from app.infrastructure.db.repositories.document_repository import DocumentRepository


@pytest.fixture()
def session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        yield session
    Base.metadata.drop_all(engine)


@pytest.fixture()
def document() -> Document:
    return Document(
        title="Repository Guide",
        source="guide.md",
        source_url=None,
        metadata={"author": "tester"},
        pages=[DocumentPage(page_number=1, content="content")],
    )


@pytest.fixture()
def chunks(document: Document) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            document_id=document.id,
            content="first chunk",
            chunk_index=0,
            page_number=1,
            metadata={"heading": "Intro"},
        ),
        DocumentChunk(
            document_id=document.id,
            content="second chunk",
            chunk_index=1,
            page_number=2,
            metadata={"heading": "Details"},
        ),
    ]


def test_save_and_fetch_document(
    session: Session, document: Document
) -> None:
    repo = DocumentRepository(session)
    repo.save_document(document)
    session.commit()

    loaded = repo.get_document(document.id)

    assert loaded is not None
    assert loaded.title == "Repository Guide"
    assert loaded.source == "guide.md"
    assert loaded.metadata_ == {"author": "tester"}


def test_save_document_is_idempotent(session: Session, document: Document) -> None:
    repo = DocumentRepository(session)
    repo.save_document(document)
    session.commit()

    document.title = "Renamed"
    repo.save_document(document)
    session.commit()

    assert repo.count_documents() == 1
    loaded = repo.get_document(document.id)
    assert loaded is not None
    assert loaded.title == "Renamed"


def test_upsert_chunks_and_reingest(
    session: Session, document: Document, chunks: list[DocumentChunk]
) -> None:
    repo = DocumentRepository(session)
    repo.save_document(document)
    repo.upsert_chunks(chunks, embeddings=[[1.0], [2.0]])
    session.commit()

    stored = repo.get_chunks(document.id)
    assert [chunk.content for chunk in stored] == ["first chunk", "second chunk"]
    assert [chunk.embedding for chunk in stored] == [[1.0], [2.0]]

    chunks[0].content = "rewritten chunk"
    repo.upsert_chunks(chunks, embeddings=[[9.0], [2.0]])
    session.commit()

    stored = repo.get_chunks(document.id)
    assert len(stored) == 2
    assert stored[0].content == "rewritten chunk"
    assert stored[0].embedding == [9.0]


def test_upsert_chunks_requires_aligned_embeddings(
    session: Session, document: Document, chunks: list[DocumentChunk]
) -> None:
    repo = DocumentRepository(session)
    repo.save_document(document)

    with pytest.raises(ValueError, match="align"):
        repo.upsert_chunks(chunks, embeddings=[[1.0]])


def test_list_and_count_documents(
    session: Session, document: Document
) -> None:
    repo = DocumentRepository(session)
    repo.save_document(document)

    duplicate = Document(
        title="Another",
        source="another.md",
        pages=[DocumentPage(page_number=1, content="content")],
    )
    repo.save_document(duplicate)
    session.commit()

    assert repo.count_documents() == 2
    assert {doc.source for doc in repo.list_documents()} == {
        "guide.md",
        "another.md",
    }


def test_get_document_by_source(session: Session, document: Document) -> None:
    repo = DocumentRepository(session)
    repo.save_document(document)
    session.commit()

    found = repo.get_document_by_source("guide.md")

    assert found is not None
    assert found.id == document.id


def test_delete_document(session: Session, document: Document) -> None:
    repo = DocumentRepository(session)
    repo.save_document(document)
    session.commit()

    assert repo.delete_document(document.id) is True
    assert repo.delete_document(uuid4()) is False
    assert repo.count_documents() == 0


def test_concrete_repository_implements_domain_contract() -> None:
    for method in ("save_document", "upsert_chunks", "persist"):
        assert hasattr(DocumentRepository, method)
        assert hasattr(DocumentRepositoryProtocol, method)


def test_persist_round_trips_document_chunks_and_embeddings(
    session: Session, document: Document, chunks: list[DocumentChunk]
) -> None:
    repo = DocumentRepository(session)
    repo.persist(document, chunks, embeddings=[[1.0], [2.0]])
    session.commit()

    assert repo.get_document(document.id) is not None
    stored = repo.get_chunks(document.id)
    assert [(chunk.content, chunk.embedding) for chunk in stored] == [
        ("first chunk", [1.0]),
        ("second chunk", [2.0]),
    ]


def test_persist_is_idempotent_and_prunes_stale_chunks(
    session: Session, document: Document, chunks: list[DocumentChunk]
) -> None:
    repo = DocumentRepository(session)
    repo.persist(document, chunks, embeddings=[[1.0], [2.0]])
    session.commit()

    chunks[0].content = "rewritten chunk"
    repo.persist(document, chunks[:1], embeddings=[[9.0]])
    session.commit()

    stored = repo.get_chunks(document.id)
    assert [(chunk.content, chunk.embedding) for chunk in stored] == [
        ("rewritten chunk", [9.0]),
    ]


def test_persist_rolls_back_atomically_on_error(
    session: Session, document: Document, chunks: list[DocumentChunk]
) -> None:
    repo = DocumentRepository(session)

    with pytest.raises(ValueError, match="align"):
        repo.persist(document, chunks, embeddings=[[1.0]])
    session.rollback()

    assert repo.count_documents() == 0
    assert repo.get_chunks(document.id) == []