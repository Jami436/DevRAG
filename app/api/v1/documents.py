import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from app.domain.documents.entities import Document
from app.infrastructure.db.models import ChunkModel, DocumentModel
from app.infrastructure.ingestion.parser_registry import UnknownFileTypeError
from app.services.documents import DocumentService, build_default_document_service

router = APIRouter()


class DocumentResponse(BaseModel):
    id: UUID
    title: str
    source: str
    source_url: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime


class DocumentChunkResponse(BaseModel):
    id: UUID
    document_id: UUID
    content: str
    chunk_index: int
    page_number: int
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentDetailResponse(DocumentResponse):
    chunks: list[DocumentChunkResponse] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse] = Field(default_factory=list)
    total: int
    limit: int
    offset: int


@lru_cache
def get_document_service() -> DocumentService:
    """Build the default document service once per process."""
    return build_default_document_service()


def _ingested_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        title=document.title,
        source=document.source,
        source_url=document.source_url,
        metadata=dict(document.metadata),
        created_at=document.created_at,
    )


def _stored_response(model: DocumentModel) -> DocumentResponse:
    return DocumentResponse(
        id=model.id,
        title=model.title,
        source=model.source,
        source_url=model.source_url,
        metadata=dict(model.metadata_),
        created_at=model.created_at,
    )


def _chunk_response(model: ChunkModel) -> DocumentChunkResponse:
    return DocumentChunkResponse(
        id=model.id,
        document_id=model.document_id,
        content=model.content,
        chunk_index=model.chunk_index,
        page_number=model.page_number,
        metadata=dict(model.metadata_),
    )


@router.post(
    "/documents",
    tags=["Documents"],
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Ingest an uploaded documentation file and index it for retrieval."""
    filename = Path(file.filename or "document.md").name
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / filename
        target.write_bytes(file.file.read())
        try:
            document = document_service.ingest(target)
        except UnknownFileTypeError as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(exc),
            ) from exc
    return _ingested_response(document)


@router.get("/documents", tags=["Documents"])
def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """List indexed documents, newest first, with pagination metadata."""
    items, total = document_service.list(limit=limit, offset=offset)
    return DocumentListResponse(
        items=[_stored_response(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/documents/{document_id}", tags=["Documents"])
def get_document(
    document_id: UUID,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentDetailResponse:
    """Return a document's metadata together with its indexed chunks."""
    document = document_service.get(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    response = _stored_response(document)
    chunks = document_service.get_chunks(document_id)
    return DocumentDetailResponse(
        id=response.id,
        title=response.title,
        source=response.source,
        source_url=response.source_url,
        metadata=response.metadata,
        created_at=response.created_at,
        chunks=[_chunk_response(chunk) for chunk in chunks],
    )


@router.delete(
    "/documents/{document_id}",
    tags=["Documents"],
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: UUID,
    document_service: DocumentService = Depends(get_document_service),
) -> None:
    """Delete a document and all of its indexed chunks."""
    if not document_service.delete(document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )