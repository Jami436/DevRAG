from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.ingestion.ingest_and_index_document import (
    IngestAndIndexDocument,
)
from app.core.settings import Settings
from app.domain.documents.chunks import DocumentChunk
from app.domain.documents.entities import Document, DocumentPage
from app.domain.documents.interfaces import DocumentParser
from app.infrastructure.db.base import Base
from app.infrastructure.db.repositories.document_repository import DocumentRepository
from app.infrastructure.ingestion.parser_registry import (
    ParserRegistry,
    build_default_parser_registry,
)
from app.infrastructure.ingestion.pipeline import (
    UnknownChunkingStrategyError,
    build_chunker,
    build_default_ingestion_pipeline,
)
from app.infrastructure.ingestion.section_chunker import SectionChunker
from app.infrastructure.ingestion.token_chunker import TokenChunker


class _RecordingEmbeddingProvider:
    """Embedding provider stub that records inputs and returns fixed vectors."""

    def __init__(self) -> None:
        self.inputs: list[list[str]] = []
        self.vectors: dict[str, list[float]] = {}

    @property
    def dimension(self) -> int:
        return 1

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.inputs.append(texts)
        return [self.vectors.get(text, [1.0]) for text in texts]


class _RecordingRepository:
    """Repository stub that records every call it receives."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.persisted: (
            tuple[Document, list[DocumentChunk], list[list[float]] | None] | None
        ) = None
        self.commits = 0
        self.closed = 0
        self.fail_on_persist = False

    def save_document(self, document: Document) -> None:
        self.events.append("save_document")

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self.events.append("upsert_chunks")

    def persist(
        self,
        document: Document,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self.events.append("persist")
        self.persisted = (document, chunks, embeddings)
        if self.fail_on_persist:
            raise RuntimeError("storage exploded")

    def commit(self) -> None:
        self.events.append("commit")
        self.commits += 1

    def close(self) -> None:
        self.events.append("close")
        self.closed += 1


class _FixedParser:
    """Parser stub returning the registry's fixed document."""

    def __init__(self, registry: "_RecordingParserRegistry") -> None:
        self._registry = registry

    def parse(self, file_path: Path) -> Document:
        self._registry.parsed_paths.append(file_path)
        return self._registry.document


class _RecordingParserRegistry:
    """Parser registry stub emitting a fixed document and recording usage."""

    def __init__(self, document: Document) -> None:
        self.document = document
        self.parsed_paths: list[Path] = []

    def get_parser(self, file_path: Path) -> DocumentParser:
        del file_path
        return _FixedParser(self)


class _RecordingChunker:
    """Chunker stub that records the document and returns given chunks."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = chunks
        self.chunked_documents: list[Document] = []

    def chunk(self, document: Document) -> list[DocumentChunk]:
        self.chunked_documents.append(document)
        return self._chunks


@pytest.fixture()
def document() -> Document:
    return Document(
        title="Pipeline Doc",
        source="pipeline.md",
        pages=[DocumentPage(page_number=1, content="content")],
    )


@pytest.fixture()
def chunks(document: Document) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            document_id=document.id,
            content="alpha",
            chunk_index=0,
            page_number=1,
        ),
        DocumentChunk(
            document_id=document.id,
            content="beta",
            chunk_index=1,
            page_number=1,
        ),
    ]


def test_execute_parses_chunks_embeds_and_stores_in_order(
    document: Document, chunks: list[DocumentChunk]
) -> None:
    order: list[str] = []

    class OrderedEmbedder(_RecordingEmbeddingProvider):
        def embed(self, texts: list[str]) -> list[list[float]]:
            order.append("embed")
            return super().embed(texts)

    class OrderedParserRegistry(_RecordingParserRegistry):
        def get_parser(self, file_path: Path) -> DocumentParser:
            order.append("parse")
            return super().get_parser(file_path)

    class OrderedChunker(_RecordingChunker):
        def chunk(self, doc: Document) -> list[DocumentChunk]:
            order.append("chunk")
            return super().chunk(doc)

    class OrderedRepository(_RecordingRepository):
        def persist(
            self,
            doc: Document,
            chunk_list: list[DocumentChunk],
            embeddings: list[list[float]] | None = None,
        ) -> None:
            order.append("store")
            return super().persist(doc, chunk_list, embeddings)

    pipeline = IngestAndIndexDocument(
        parser_registry=OrderedParserRegistry(document),
        chunker=OrderedChunker(chunks),
        embedding_provider=OrderedEmbedder(),
        repository_factory=lambda: OrderedRepository(),
    )

    result = pipeline.execute(Path("guide.md"))

    assert order == ["parse", "chunk", "embed", "store"]
    assert result is document


def test_execute_embeds_chunk_contents_and_stores_them(
    document: Document, chunks: list[DocumentChunk]
) -> None:
    surviving = [chunks[1]]  # only the "beta" chunk survives chunking
    embedder = _RecordingEmbeddingProvider()
    embedder.vectors["beta"] = [0.5]
    repository = _RecordingRepository()

    pipeline = IngestAndIndexDocument(
        parser_registry=_RecordingParserRegistry(document),
        chunker=_RecordingChunker(surviving),
        embedding_provider=embedder,
        repository_factory=lambda: repository,
    )

    pipeline.execute(Path("guide.md"))

    assert embedder.inputs == [["beta"]]
    assert repository.persisted == (document, surviving, [[0.5]])
    assert repository.closed == 1


def test_execute_commits_and_closes_repository(
    document: Document, chunks: list[DocumentChunk]
) -> None:
    repository = _RecordingRepository()

    pipeline = IngestAndIndexDocument(
        parser_registry=_RecordingParserRegistry(document),
        chunker=_RecordingChunker(chunks),
        embedding_provider=_RecordingEmbeddingProvider(),
        repository_factory=lambda: repository,
    )

    pipeline.execute(Path("guide.md"))

    assert repository.commits == 1
    assert repository.closed == 1
    assert repository.events[-2:] == ["commit", "close"]


def test_execute_closes_repository_when_persist_fails(
    document: Document, chunks: list[DocumentChunk]
) -> None:
    repository = _RecordingRepository()
    repository.fail_on_persist = True

    pipeline = IngestAndIndexDocument(
        parser_registry=_RecordingParserRegistry(document),
        chunker=_RecordingChunker(chunks),
        embedding_provider=_RecordingEmbeddingProvider(),
        repository_factory=lambda: repository,
    )

    with pytest.raises(RuntimeError, match="storage exploded"):
        pipeline.execute(Path("guide.md"))

    assert repository.closed == 1
    assert repository.commits == 0


def test_default_pipeline_is_wired_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

    pipeline = build_default_ingestion_pipeline()

    assert isinstance(pipeline._parser_registry, ParserRegistry)
    assert isinstance(pipeline._chunker, SectionChunker)
    assert isinstance(pipeline._embedding_provider, OpenAIEmbeddingProvider)


@pytest.mark.parametrize(
    ("strategy", "expected_type"),
    [
        ("section", SectionChunker),
        ("token", TokenChunker),
    ],
)
def test_build_chunker_selects_strategy(
    strategy: str, expected_type: type[Any]
) -> None:
    app_settings = Settings(
        chunking_strategy=strategy,
        chunking_token_limit=64,
        chunking_token_overlap=8,
    )

    chunker = build_chunker(app_settings)

    assert isinstance(chunker, expected_type)


def test_build_chunker_rejects_unknown_strategy() -> None:
    app_settings = Settings(chunking_strategy="paragraph")

    with pytest.raises(UnknownChunkingStrategyError, match="paragraph"):
        build_chunker(app_settings)


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


def test_pipeline_end_to_end_ingests_sample_markdown(
    sample_markdown: Path, session_factory: sessionmaker[Session]
) -> None:
    embedder = _RecordingEmbeddingProvider()
    pipeline = IngestAndIndexDocument(
        parser_registry=build_default_parser_registry(),
        chunker=TokenChunker(token_limit=20),
        embedding_provider=embedder,
        repository_factory=lambda: DocumentRepository(session_factory()),
    )

    document = pipeline.execute(sample_markdown)

    assert document.title == "Iris KNN Classification"
    with session_factory() as inspection_session:
        repo = DocumentRepository(inspection_session)
        stored = repo.get_chunks(document.id)
        assert len(stored) > 0
        assert stored[0].embedding == [1.0]
    assert embedder.inputs == [[chunk.content for chunk in stored]]