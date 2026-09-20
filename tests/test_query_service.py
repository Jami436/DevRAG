from uuid import uuid4

import pytest

from app.domain.generation.interfaces import LLMProvider
from app.domain.retrieval.entities import RetrievedChunk
from app.services.query import QueryService, build_default_query_service


class _FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _text in texts]


class _FakeSearchRepository:
    def __init__(self) -> None:
        self.closed = False
        self.results: list[RetrievedChunk] = []

    def keyword_search(
        self, query_text: str, *, limit: int = 10
    ) -> list[RetrievedChunk]:
        return []

    def hybrid_search(self, **kwargs: object) -> list[RetrievedChunk]:
        return self.results

    def close(self) -> None:
        self.closed = True


class _Reranker:
    def rerank(
        self, query: str, results: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        for rank, hit in enumerate(results, start=1):
            hit.rank = rank
            hit.final_score = rank
        return list(results)


class _Provider(LLMProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return "pytest-llm"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        del max_tokens
        self.calls.append((system_prompt, user_prompt))
        return "The answer comes from the cited sources [1]."


def _hit() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="vector databases rank by similarity",
        chunk_index=0,
        page_number=1,
        document_title="Guide",
    )


def test_execute_retrieves_then_generates_with_citations() -> None:
    repo = _FakeSearchRepository()
    repo.results = [_hit(), _hit()]
    provider = _Provider()

    service = QueryService(
        embedding_provider=_FakeEmbedder(),
        search_repository_factory=lambda: repo,
        reranker=_Reranker(),
        llm_provider=provider,
        top_k=2,
        candidates=10,
    )

    answer = service.execute("what is a vector database?")

    assert len(provider.calls) == 1
    assert answer.answer == "The answer comes from the cited sources [1]."
    assert answer.model == "pytest-llm"
    assert [citation.chunk_id for citation in answer.citations] == [
        repo.results[0].chunk_id
    ]
    assert repo.closed is True


def test_execute_truncates_retrieval_to_requested_top_k() -> None:
    repo = _FakeSearchRepository()
    repo.results = [_hit() for _ in range(5)]
    provider = _Provider()

    service = QueryService(
        embedding_provider=_FakeEmbedder(),
        search_repository_factory=lambda: repo,
        reranker=_Reranker(),
        llm_provider=provider,
        top_k=5,
        candidates=10,
    )

    answer = service.execute("question", top_k=2)

    assert "[1]" in provider.calls[0][1]
    assert "[2]" in provider.calls[0][1]
    assert "[3]" not in provider.calls[0][1]
    assert answer.query == "question"


def test_execute_uses_default_top_k_when_not_overridden() -> None:
    repo = _FakeSearchRepository()
    repo.results = [_hit() for _ in range(3)]
    provider = _Provider()

    service = QueryService(
        embedding_provider=_FakeEmbedder(),
        search_repository_factory=lambda: repo,
        reranker=_Reranker(),
        llm_provider=provider,
        top_k=3,
        candidates=10,
    )

    service.execute("question")

    assert "[3]" in provider.calls[0][1]


def test_execute_rejects_non_positive_top_k() -> None:
    service = QueryService(
        embedding_provider=_FakeEmbedder(),
        search_repository_factory=lambda: _FakeSearchRepository(),
        reranker=_Reranker(),
        llm_provider=_Provider(),
    )

    with pytest.raises(ValueError, match="top_k"):
        service.execute("question", top_k=0)


def test_default_query_service_is_wired_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider
    from app.infrastructure.llm.openai_provider import OpenAILLMProvider
    from app.infrastructure.reranking.cross_encoder_reranker import (
        CrossEncoderReranker,
    )

    service = build_default_query_service()

    assert isinstance(service._embedding_provider, OpenAIEmbeddingProvider)
    assert isinstance(service._reranker, CrossEncoderReranker)
    assert isinstance(service._llm_provider, OpenAILLMProvider)
    assert service._top_k == 5
    assert service._candidates == 50