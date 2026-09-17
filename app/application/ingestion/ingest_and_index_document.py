from collections.abc import Callable
from pathlib import Path

from app.domain.documents.chunks import DocumentChunk
from app.domain.documents.entities import Document
from app.domain.documents.interfaces import Chunker, ParserRegistry
from app.domain.embeddings.interfaces import EmbeddingProvider
from app.domain.repositories.interfaces import DocumentRepository


class IngestAndIndexDocument:
    """Run a document through the full ingestion pipeline end to end.

    Composition of the four stages — parse, chunk, embed, store — relying on
    injected ports so every stage can be swapped or faked independently.
    """

    def __init__(
        self,
        parser_registry: ParserRegistry,
        chunker: Chunker,
        embedding_provider: EmbeddingProvider,
        repository_factory: Callable[[], DocumentRepository],
    ) -> None:
        self._parser_registry = parser_registry
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._repository_factory = repository_factory

    def execute(self, file_path: Path) -> Document:
        document = self._parse(file_path)
        chunks = self._chunker.chunk(document)
        embeddings = self._embedding_provider.embed(
            [chunk.content for chunk in chunks]
        )
        self._store(document, chunks, embeddings)
        return document

    def _parse(self, file_path: Path) -> Document:
        parser = self._parser_registry.get_parser(file_path)
        return parser.parse(file_path)

    def _store(
        self,
        document: Document,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        repository = self._repository_factory()
        try:
            repository.persist(document, chunks, embeddings)
            repository.commit()
        finally:
            repository.close()