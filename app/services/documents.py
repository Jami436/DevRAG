from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from uuid import UUID

from app.domain.documents.entities import Document
from app.infrastructure.db.models import ChunkModel, DocumentModel
from app.infrastructure.db.repositories.document_repository import (
    DocumentRepository as SqlDocumentRepository,
)
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.ingestion.pipeline import build_default_ingestion_pipeline


class IngestionPipeline(Protocol):
    """Ingest a source file end to end and return the indexed document."""

    def execute(self, file_path: Path) -> Document: ...


class DocumentManager(Protocol):
    """Read surface of the document store the service depends on.

    The concrete SQL repository already implements these operations with
    storage-specific model return types; the protocol lets the service operate
    against fakes in tests and alternative stores in the future.
    """

    def get_document(self, document_id: UUID) -> DocumentModel | None: ...

    def get_chunks(self, document_id: UUID) -> list[ChunkModel]: ...

    def list_documents(self, limit: int, offset: int) -> list[DocumentModel]: ...

    def count_documents(self) -> int: ...

    def delete_document(self, document_id: UUID) -> bool: ...

    def commit(self) -> None: ...

    def close(self) -> None: ...


class DocumentService:
    """Manage the lifecycle of indexed documents.

    Ingests new sources through the full parse -> chunk -> embed -> store
    pipeline and exposes storage-backed lookups (get, list, delete) on top of a
    fresh session per call so results are never surfaced from a closed session.
    """

    def __init__(
        self,
        ingestion_pipeline: IngestionPipeline,
        document_repository_factory: Callable[[], DocumentManager],
    ) -> None:
        self._pipeline = ingestion_pipeline
        self._repository_factory = document_repository_factory

    def ingest(self, file_path: Path) -> Document:
        """Parse, chunk, embed and persist the document at ``file_path``."""
        return self._pipeline.execute(file_path)

    def get(self, document_id: UUID) -> DocumentModel | None:
        repository = self._repository_factory()
        try:
            return repository.get_document(document_id)
        finally:
            repository.close()

    def get_chunks(self, document_id: UUID) -> list[ChunkModel]:
        repository = self._repository_factory()
        try:
            return repository.get_chunks(document_id)
        finally:
            repository.close()

    def list(self, *, limit: int, offset: int) -> tuple[list[DocumentModel], int]:
        """Return a page of documents together with the total document count."""
        repository = self._repository_factory()
        try:
            return (
                repository.list_documents(limit=limit, offset=offset),
                repository.count_documents(),
            )
        finally:
            repository.close()

    def delete(self, document_id: UUID) -> bool:
        """Delete a document and return whether it previously existed."""
        repository = self._repository_factory()
        try:
            deleted = repository.delete_document(document_id)
            if deleted:
                repository.commit()
            return deleted
        finally:
            repository.close()


def build_default_document_service() -> DocumentService:
    """Wire the default document service from application settings."""
    return DocumentService(
        ingestion_pipeline=build_default_ingestion_pipeline(),
        document_repository_factory=_document_repository_factory,
    )


def _document_repository_factory() -> DocumentManager:
    return SqlDocumentRepository(SessionLocal())