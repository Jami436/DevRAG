from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.documents.entities import Document, DocumentPage
from app.infrastructure.db.base import Base
from app.infrastructure.db.models import ChunkModel, DocumentModel


@pytest.fixture()
def db_engine() -> Any:
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
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture()
def session(db_engine: Any) -> Iterator[Session]:
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    with factory() as session:
        yield session


def test_document_model_round_trips_metadata(session: Session) -> None:
    document = Document(
        title="Use Case",
        source="example.md",
        source_url="https://example.com/use-case",
        metadata={"author": "tester"},
        pages=[DocumentPage(page_number=1, content="content")],
    )

    saved = DocumentModel(
        id=document.id,
        title=document.title,
        source=document.source,
        source_url=document.source_url,
        metadata_=dict(document.metadata),
        created_at=document.created_at,
    )
    session.add(saved)
    session.commit()

    loaded = session.get(DocumentModel, document.id)
    assert loaded is not None
    assert loaded.title == "Use Case"
    assert loaded.source == "example.md"
    assert loaded.source_url == "https://example.com/use-case"
    assert loaded.metadata_ == {"author": "tester"}
    assert loaded.created_at is not None


def test_chunk_model_round_trips_embedding(session: Session) -> None:
    document = DocumentModel(
        id=UUID("11111111-1111-1111-1111-111111111111"),
        title="Doc",
        source="doc.md",
    )
    chunk = ChunkModel(
        id=UUID("22222222-2222-2222-2222-222222222222"),
        document_id=document.id,
        content="hello world",
        chunk_index=0,
        page_number=1,
        metadata_={"heading": "Intro"},
        embedding=[0.1, 0.2, 0.3],
    )
    session.add_all([document, chunk])
    session.commit()

    loaded = session.scalar(select(ChunkModel))
    assert loaded is not None
    assert loaded.content == "hello world"
    assert loaded.metadata_ == {"heading": "Intro"}
    assert loaded.embedding == [0.1, 0.2, 0.3]


def test_chunk_requires_unique_document_index_pair(session: Session) -> None:
    document = DocumentModel(id=uuid4(), title="Doc", source="doc.md")
    session.add(document)
    session.flush()
    session.add(
        ChunkModel(
            document_id=document.id,
            content="first",
            chunk_index=0,
            page_number=1,
        )
    )
    session.add(
        ChunkModel(
            document_id=document.id,
            content="second",
            chunk_index=0,
            page_number=2,
        )
    )

    with pytest.raises(IntegrityError):
        session.commit()


def test_deleting_document_cascades_to_chunks(session: Session) -> None:
    document = DocumentModel(id=uuid4(), title="Doc", source="doc.md")
    session.add(document)
    session.flush()
    session.add(
        ChunkModel(
            document_id=document.id,
            content="only chunk",
            chunk_index=0,
            page_number=1,
        )
    )
    session.commit()

    session.delete(session.get(DocumentModel, document.id))
    session.commit()

    chunk_count = session.scalar(select(func.count(ChunkModel.id))) or 0
    assert chunk_count == 0