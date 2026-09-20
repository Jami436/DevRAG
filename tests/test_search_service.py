from uuid import uuid4

import pytest

from app.core.settings import Settings
from app.domain.retrieval.entities import RetrievedChunk
from app.services.search import (
    SearchService,
    build_default_search_service,
)


class _FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _text in texts]


class _FakeSearchRepository:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[tuple[str, list[float], int]] = []
        self.results: list[RetrievedChunk] = []

    def keyword_search(
        self, query_text: str, *, limit: int = 10
    ) -> list[RetrievedChunk]:
        return []

    def hybrid_search(
        self,
        *,
        query_text: str,
        query_embedding: list[float],
        limit: int = 10,
        candidates: int = 50,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        fusion_k: int = 60,
        distance_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        del candidates, vector_weight, keyword_weight, fusion_k, distance_threshold
        self.calls.append((query_text, query_embedding, limit))
        return self.results

    def close(self) -> None:
        self.closed = True


class _Reranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def rerank(
        self, query: str, results: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        self.calls.append((query, len(results)))
        for rank, hit in enumerate(results, start=1):
            hit.rank = rank
            hit.final_score = rank
        return list(results)


def _hit() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="vector databases rank by similarity",
        chunk_index=0,
        page_number=1,
        document_title="Guide",
    )


def test_execute_retrieves_and_reranks() -> None:
    repo = _FakeSearchRepository()
    repo.results = [_hit() for _ in range(5)]
    reranker = _Reranker()
    service = SearchService(
        embedding_provider=_FakeEmbedder(),
        search_repository_factory=lambda: repo,
        reranker=reranker,
        top_k=3,
        candidates=20,
    )

    results = service.execute("what is a vector database?")

    assert len(results) == 3
    assert repo.closed is True
    assert repo.calls == [
        ("what is a vector database?", [1.0, 0.0, 0.0], 20)
    ]
    assert reranker.calls == [("what is a vector database?", 5)]
    assert [hit.rank for hit in results] == [1, 2, 3]


def test_execute_uses_requested_top_k() -> None:
    repo = _FakeSearchRepository()
    repo.results = [_hit() for _ in range(5)]
    service = SearchService(
        embedding_provider=_FakeEmbedder(),
        search_repository_factory=lambda: repo,
        reranker=_Reranker(),
        top_k=5,
        candidates=10,
    )

    results = service.execute("question", top_k=2)

    assert len(results) == 2


def test_execute_uses_configured_top_k_by_default() -> None:
    repo = _FakeSearchRepository()
    repo.results = [_hit() for _ in range(5)]
    service = SearchService(
        embedding_provider=_FakeEmbedder(),
        search_repository_factory=lambda: repo,
        reranker=_Reranker(),
        top_k=4,
        candidates=10,
    )

    results = service.execute("question")

    assert len(results) == 4


def test_execute_rejects_non_positive_top_k() -> None:
    service = SearchService(
        embedding_provider=_FakeEmbedder(),
        search_repository_factory=lambda: _FakeSearchRepository(),
        reranker=_Reranker(),
    )

    with pytest.raises(ValueError, match="top_k"):
        service.execute("question", top_k=0)


def test_default_search_service_is_wired_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    app_settings = Settings(
        retriever_top_k=7,
        retrieval_hybrid_candidates=99,
    )
    from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider
    from app.infrastructure.reranking.cross_encoder_reranker import (
        CrossEncoderReranker,
    )

    service = build_default_search_service(app_settings)

    assert isinstance(service._embedding_provider, OpenAIEmbeddingProvider)
    assert isinstance(service._reranker, CrossEncoderReranker)
    assert service._top_k == 7
    assert service._candidates == 99