from uuid import uuid4

import pytest

from app.application.retrieval.retrieve_context import RetrieveContext
from app.domain.retrieval.entities import RetrievedChunk


class _FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _text in texts]


class _FakeSearchRepository:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False
        self.results: list[RetrievedChunk] = []

    def keyword_search(
        self, query_text: str, *, limit: int = 10
    ) -> list[RetrievedChunk]:
        return []

    def hybrid_search(self, **kwargs: object) -> list[RetrievedChunk]:
        self.calls.append(kwargs)
        return self.results

    def close(self) -> None:
        self.closed = True


class _RecordingReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def rerank(
        self, query: str, results: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        self.calls.append((query, len(results)))
        for rank, hit in enumerate(results, start=1):
            hit.rank = rank
        return list(results)


def _hit() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="content",
        chunk_index=0,
        page_number=1,
    )


def test_execute_embeds_hybrid_searches_and_reranks() -> None:
    repo = _FakeSearchRepository()
    repo.results = [_hit() for _ in range(5)]
    reranker = _RecordingReranker()
    use_case = RetrieveContext(
        embedding_provider=_FakeEmbedder(),
        search_repository_factory=lambda: repo,
        reranker=reranker,
        top_k=3,
        candidates=10,
    )

    results = use_case.execute("what is a vector database?")

    assert len(results) == 3
    assert repo.closed is True
    assert repo.calls == [
        {
            "query_text": "what is a vector database?",
            "query_embedding": [1.0, 0.0, 0.0],
            "limit": 10,
        }
    ]
    assert reranker.calls == [("what is a vector database?", 5)]


def test_execute_returns_all_when_fewer_than_top_k() -> None:
    repo = _FakeSearchRepository()
    repo.results = [_hit()]
    use_case = RetrieveContext(
        embedding_provider=_FakeEmbedder(),
        search_repository_factory=lambda: repo,
        reranker=_RecordingReranker(),
        top_k=5,
        candidates=5,
    )

    results = use_case.execute("short query")

    assert len(results) == 1


def test_execute_closes_repository_even_on_reranker_failure() -> None:
    class _BoomReranker:
        def rerank(
            self, query: str, results: list[RetrievedChunk]
        ) -> list[RetrievedChunk]:
            raise RuntimeError("rerank failed")

    repo = _FakeSearchRepository()
    use_case = RetrieveContext(
        embedding_provider=_FakeEmbedder(),
        search_repository_factory=lambda: repo,
        reranker=_BoomReranker(),
    )

    with pytest.raises(RuntimeError, match="rerank failed"):
        use_case.execute("boom")

    assert repo.closed is True


def test_execute_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        RetrieveContext(
            embedding_provider=_FakeEmbedder(),
            search_repository_factory=lambda: _FakeSearchRepository(),
            reranker=_RecordingReranker(),
            top_k=0,
        )


def test_execute_rejects_candidates_below_top_k() -> None:
    with pytest.raises(ValueError, match="candidates"):
        RetrieveContext(
            embedding_provider=_FakeEmbedder(),
            search_repository_factory=lambda: _FakeSearchRepository(),
            reranker=_RecordingReranker(),
            top_k=10,
            candidates=5,
        )