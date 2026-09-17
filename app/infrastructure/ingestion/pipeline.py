from app.application.ingestion.ingest_and_index_document import IngestAndIndexDocument
from app.core.settings import Settings, settings
from app.domain.documents.interfaces import Chunker
from app.domain.repositories.interfaces import DocumentRepository
from app.infrastructure.db.repositories.document_repository import (
    DocumentRepository as SqlDocumentRepository,
)
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.embeddings.factory import build_embedding_provider
from app.infrastructure.ingestion.parser_registry import (
    build_default_parser_registry,
)
from app.infrastructure.ingestion.section_chunker import SectionChunker
from app.infrastructure.ingestion.token_chunker import TokenChunker


class UnknownChunkingStrategyError(ValueError):
    """Raised when no chunker matches the configured chunking strategy."""


def build_chunker(app_settings: Settings) -> Chunker:
    """Build the chunker selected by ``app_settings.chunking_strategy``."""
    strategy = app_settings.chunking_strategy
    if strategy == "section":
        return SectionChunker(token_limit=app_settings.chunking_token_limit)
    if strategy == "token":
        return TokenChunker(
            token_limit=app_settings.chunking_token_limit,
            token_overlap=app_settings.chunking_token_overlap,
        )
    raise UnknownChunkingStrategyError(
        f"Unknown chunking strategy: {strategy}"
    )


def build_default_ingestion_pipeline() -> IngestAndIndexDocument:
    """Wire the default ingestion pipeline from application settings."""
    return IngestAndIndexDocument(
        parser_registry=build_default_parser_registry(),
        chunker=build_chunker(settings),
        embedding_provider=build_embedding_provider(settings),
        repository_factory=_repository_factory,
    )


def _repository_factory() -> DocumentRepository:
    return SqlDocumentRepository(SessionLocal())