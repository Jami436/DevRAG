from collections.abc import Callable

from app.application.generation.generate_answer import GenerateAnswer
from app.application.retrieval.retrieve_context import RetrieveContext
from app.core.settings import Settings, settings
from app.domain.embeddings.interfaces import EmbeddingProvider
from app.domain.generation.entities import GeneratedAnswer
from app.domain.generation.interfaces import LLMProvider
from app.domain.repositories.interfaces import SearchRepository
from app.domain.retrieval.interfaces import Reranker
from app.infrastructure.db.repositories.document_repository import (
    DocumentRepository as SqlDocumentRepository,
)
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.embeddings.factory import build_embedding_provider
from app.infrastructure.llm.factory import build_llm_provider
from app.infrastructure.reranking.factory import build_reranker


class QueryService:
    """Configurable end-to-end question answering over indexed documents.

    Composes the retrieval, reranking, context construction and grounded
    generation stages; ``top_k`` can be narrowed per request while every other
    tuning knob is fixed at construction time.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        search_repository_factory: Callable[[], SearchRepository],
        reranker: Reranker,
        llm_provider: LLMProvider,
        top_k: int = 5,
        candidates: int = 50,
        max_excerpt_chars: int = 1000,
        max_total_chars: int = 12000,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._search_repository_factory = search_repository_factory
        self._reranker = reranker
        self._llm_provider = llm_provider
        self._top_k = top_k
        self._candidates = candidates
        self._max_excerpt_chars = max_excerpt_chars
        self._max_total_chars = max_total_chars

    def execute(self, query: str, *, top_k: int | None = None) -> GeneratedAnswer:
        effective_top_k = top_k if top_k is not None else self._top_k
        chunks = RetrieveContext(
            embedding_provider=self._embedding_provider,
            search_repository_factory=self._search_repository_factory,
            reranker=self._reranker,
            top_k=effective_top_k,
            candidates=self._candidates,
        ).execute(query)
        return GenerateAnswer(
            llm_provider=self._llm_provider,
            max_excerpt_chars=self._max_excerpt_chars,
            max_total_chars=self._max_total_chars,
        ).execute(query, chunks)


def build_default_query_service(app_settings: Settings = settings) -> QueryService:
    """Wire the default query service from application settings."""
    return QueryService(
        embedding_provider=build_embedding_provider(app_settings),
        search_repository_factory=_search_repository_factory,
        reranker=build_reranker(app_settings),
        llm_provider=build_llm_provider(app_settings),
        top_k=app_settings.retriever_top_k,
        candidates=app_settings.retrieval_hybrid_candidates,
        max_excerpt_chars=app_settings.generation_max_excerpt_chars,
        max_total_chars=app_settings.generation_max_context_chars,
    )


def _search_repository_factory() -> SearchRepository:
    return SqlDocumentRepository(SessionLocal())