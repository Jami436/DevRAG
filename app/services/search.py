from collections.abc import Callable

from app.application.retrieval.retrieve_context import RetrieveContext
from app.core.settings import Settings, settings
from app.domain.embeddings.interfaces import EmbeddingProvider
from app.domain.repositories.interfaces import SearchRepository
from app.domain.retrieval.entities import RetrievedChunk
from app.domain.retrieval.interfaces import Reranker
from app.infrastructure.db.repositories.document_repository import (
    DocumentRepository as SqlDocumentRepository,
)
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.embeddings.factory import build_embedding_provider
from app.infrastructure.reranking.factory import build_reranker


class SearchService:
    """Retrieve and rerank the most relevant chunks for a question.

    Wraps the embed, hybrid search and rerank stages so both the raw search
    endpoint and the answer-generation service share one construction site and
    one set of tuning knobs. ``top_k`` can be narrowed per request while the
    candidate pool size stays fixed at construction time.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        search_repository_factory: Callable[[], SearchRepository],
        reranker: Reranker,
        top_k: int = 5,
        candidates: int = 50,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._search_repository_factory = search_repository_factory
        self._reranker = reranker
        self._top_k = top_k
        self._candidates = candidates

    def execute(
        self, query: str, *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        effective_top_k = top_k if top_k is not None else self._top_k
        return RetrieveContext(
            embedding_provider=self._embedding_provider,
            search_repository_factory=self._search_repository_factory,
            reranker=self._reranker,
            top_k=effective_top_k,
            candidates=self._candidates,
        ).execute(query)


def build_default_search_service(app_settings: Settings = settings) -> SearchService:
    """Wire the default search service from application settings."""
    return SearchService(
        embedding_provider=build_embedding_provider(app_settings),
        search_repository_factory=_search_repository_factory,
        reranker=build_reranker(app_settings),
        top_k=app_settings.retriever_top_k,
        candidates=app_settings.retrieval_hybrid_candidates,
    )


def _search_repository_factory() -> SearchRepository:
    return SqlDocumentRepository(SessionLocal())