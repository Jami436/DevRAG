from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1.documents import get_document_service
from app.application.ingestion.ingest_and_index_document import (
    IngestAndIndexDocument,
)
from app.domain.documents.entities import Document, DocumentPage
from app.infrastructure.db.base import Base
from app.infrastructure.db.models import ChunkModel, DocumentModel
from app.infrastructure.db.repositories.document_repository import (
    DocumentRepository as SqlDocumentRepository,
)
from app.infrastructure.ingestion.parser_registry import (
    UnknownFileTypeError,
    build_default_parser_registry,
)
from app.infrastructure.ingestion.token_chunker import TokenChunker
from app.main import app
from app.services.documents import DocumentService


class _FakeDocumentService:
    def __init__(self) -> None:
        self.ingest_calls: list[Path] = []
        self.ingest_result: Document | None = None
        self.ingest_error: Exception | None = None
        self.get_results: dict[UUID, DocumentModel | None] = {}
        self.chunks: dict[UUID, list[ChunkModel]] = {}
        self.documents: list[DocumentModel] = []
        self.total = 0
        self.delete_results: dict[UUID, bool] = {}
        self.list_calls: list[tuple[int, int]] = []

    def ingest(self, file_path: Path) -> Document:
        self.ingest_calls.append(file_path)
        if self.ingest_error is not None:
            raise self.ingest_error
        assert self.ingest_result is not None
        return self.ingest_result

    def get(self, document_id: UUID) -> DocumentModel | None:
        return self.get_results.get(document_id)

    def get_chunks(self, document_id: UUID) -> list[ChunkModel]:
        return self.chunks.get(document_id, [])

    def list(self, *, limit: int, offset: int) -> tuple[list[DocumentModel], int]:
        self.list_calls.append((limit, offset))
        return self.documents, self.total

    def delete(self, document_id: UUID) -> bool:
        return self.delete_results.get(document_id, False)


def _model() -> DocumentModel:
    return DocumentModel(
        id=uuid4(),
        title="Guide",
        source="guide.md",
        source_url="https://example.com/guide",
        metadata_={"heading": "Intro"},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _chunk() -> ChunkModel:
    return ChunkModel(
        id=uuid4(),
        document_id=uuid4(),
        content="chunk content",
        chunk_index=0,
        page_number=1,
        metadata_={"heading": "Intro"},
    )


def _ingested() -> Document:
    return Document(
        id=uuid4(),
        title="Iris KNN Classification",
        source="iris.md",
        metadata={"topic": "ml"},
        pages=[DocumentPage(page_number=1, content="content")],
        created_at=datetime(2026, 2, 1, tzinfo=UTC),
    )


@pytest.fixture()
def documents_endpoint() -> Iterator[tuple[TestClient, _FakeDocumentService]]:
    fake = _FakeDocumentService()
    app.dependency_overrides[get_document_service] = lambda: fake
    with TestClient(app) as client:
        yield client, fake
    app.dependency_overrides.clear()


def _upload(client: TestClient, filename: str, content: bytes) -> Any:
    return client.post(
        "/api/v1/documents",
        files={"file": (filename, content, "application/octet-stream")},
    )


def test_upload_document_returns_created_document(
    documents_endpoint: tuple[TestClient, _FakeDocumentService],
) -> None:
    client, fake = documents_endpoint
    fake.ingest_result = _ingested()

    response = _upload(client, "iris.md", b"# Title\n\ncontent")

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Iris KNN Classification"
    assert body["metadata"] == {"topic": "ml"}
    assert isinstance(body["id"], str)
    assert Path(fake.ingest_calls[0]).name == "iris.md"


def test_upload_document_rejects_unsupported_file_type(
    documents_endpoint: tuple[TestClient, _FakeDocumentService],
) -> None:
    client, fake = documents_endpoint
    fake.ingest_error = UnknownFileTypeError("Unsupported file type: .xyz")

    response = _upload(client, "file.xyz", b"x")

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported file type: .xyz"


def test_get_document_returns_metadata_and_chunks(
    documents_endpoint: tuple[TestClient, _FakeDocumentService],
) -> None:
    client, fake = documents_endpoint
    model = _model()
    chunk = _chunk()
    fake.get_results[model.id] = model
    fake.chunks[model.id] = [chunk]

    response = client.get(f"/api/v1/documents/{model.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Guide"
    assert body["source_url"] == "https://example.com/guide"
    assert body["metadata"] == {"heading": "Intro"}
    assert body["created_at"] == "2026-01-01T00:00:00Z"
    (stored_chunk,) = body["chunks"]
    assert stored_chunk["content"] == "chunk content"
    assert stored_chunk["document_id"] == str(chunk.document_id)


def test_get_document_returns_404_when_missing(
    documents_endpoint: tuple[TestClient, _FakeDocumentService],
) -> None:
    client, fake = documents_endpoint
    fake.get_results[uuid4()] = None

    response = client.get(f"/api/v1/documents/{uuid4()}")

    assert response.status_code == 404


def test_list_documents_returns_page_and_total(
    documents_endpoint: tuple[TestClient, _FakeDocumentService],
) -> None:
    client, fake = documents_endpoint
    fake.documents = [_model(), _model()]
    fake.total = 2

    response = client.get("/api/v1/documents?limit=10&offset=0")

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 2
    assert body["total"] == 2
    assert body["limit"] == 10
    assert body["offset"] == 0
    assert fake.list_calls == [(10, 0)]


def test_list_documents_uses_default_pagination(
    documents_endpoint: tuple[TestClient, _FakeDocumentService],
) -> None:
    client, fake = documents_endpoint

    response = client.get("/api/v1/documents")

    assert response.status_code == 200
    assert fake.list_calls == [(20, 0)]


def test_list_documents_rejects_invalid_pagination(
    documents_endpoint: tuple[TestClient, _FakeDocumentService],
) -> None:
    client, _fake = documents_endpoint

    assert client.get("/api/v1/documents?limit=0").status_code == 422
    assert client.get("/api/v1/documents?limit=101").status_code == 422
    assert client.get("/api/v1/documents?offset=-1").status_code == 422


def test_delete_document_returns_204(
    documents_endpoint: tuple[TestClient, _FakeDocumentService],
) -> None:
    client, fake = documents_endpoint
    document_id = uuid4()
    fake.delete_results[document_id] = True

    response = client.delete(f"/api/v1/documents/{document_id}")

    assert response.status_code == 204
    assert response.content == b""


def test_delete_document_returns_404_when_missing(
    documents_endpoint: tuple[TestClient, _FakeDocumentService],
) -> None:
    client, _fake = documents_endpoint

    response = client.delete(f"/api/v1/documents/{uuid4()}")

    assert response.status_code == 404


class _RecordingEmbeddingProvider:
    @property
    def dimension(self) -> int:
        return 1

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _text in texts]


@pytest.fixture()
def session_factory() -> Iterator[sessionmaker[Session]]:
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
    yield factory
    Base.metadata.drop_all(engine)


def test_documents_api_upload_list_detail_delete_end_to_end(
    sample_markdown: Path,
    session_factory: sessionmaker[Session],
) -> None:
    pipeline = IngestAndIndexDocument(
        parser_registry=build_default_parser_registry(),
        chunker=TokenChunker(token_limit=20),
        embedding_provider=_RecordingEmbeddingProvider(),
        repository_factory=lambda: SqlDocumentRepository(session_factory()),
    )
    service = DocumentService(
        pipeline,
        lambda: SqlDocumentRepository(session_factory()),
    )
    app.dependency_overrides[get_document_service] = lambda: service
    try:
        with TestClient(app) as client:
            with sample_markdown.open("rb") as handle:
                upload = client.post(
                    "/api/v1/documents",
                    files={
                        "file": (sample_markdown.name, handle, "text/markdown")
                    },
                )
            assert upload.status_code == 201
            document_id = upload.json()["id"]

            listing = client.get("/api/v1/documents")
            assert listing.status_code == 200
            assert listing.json()["total"] == 1

            detail = client.get(f"/api/v1/documents/{document_id}")
            assert detail.status_code == 200
            assert detail.json()["title"] == sample_markdown.stem
            assert len(detail.json()["chunks"]) > 0

            delete = client.delete(f"/api/v1/documents/{document_id}")
            assert delete.status_code == 204

            assert (
                client.get(f"/api/v1/documents/{document_id}").status_code == 404
            )
    finally:
        app.dependency_overrides.clear()