from collections.abc import Callable

from app.domain.embeddings.interfaces import EmbeddingProvider
from app.domain.repositories.interfaces import SearchRepository
from app.domain.retrieval.entities import RetrievedChunk
from app.domain.retrieval.interfaces import Reranker


class RetrieveContext:
    """Retrieve and re-rank the most relevant context chunks for a query.

    Composition of embed, hybrid search and rerank — the read-side counterpart
    of the ingestion pipeline. Each stage is injected as a port so it can be
    swapped with fakes in tests or other providers in production.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        search_repository_factory: Callable[[], SearchRepository],
        reranker: Reranker,
        top_k: int = 5,
        candidates: int = 20,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if candidates < top_k:
            raise ValueError("candidates must be at least top_k")
        self._embedding_provider = embedding_provider
        self._search_repository_factory = search_repository_factory
        self._reranker = reranker
        self._top_k = top_k
        self._candidates = candidates

    def execute(self, query: str) -> list[RetrievedChunk]:
        query_embedding = self._embedding_provider.embed([query])[0]
        repository = self._search_repository_factory()
        try:
            candidates = repository.hybrid_search(
                query_text=query,
                query_embedding=query_embedding,
                limit=self._candidates,
            )
            reranked = self._reranker.rerank(query, candidates)
        finally:
            repository.close()
        return reranked[: self._top_k]