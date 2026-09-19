import json
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from pytest import MonkeyPatch

from app.domain.retrieval.entities import RetrievedChunk
from app.infrastructure.reranking.llm_reranker import LLMReranker


class _FakeChatCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class _FakeOpenAIClient:
    def __init__(self, content: str) -> None:
        self.completions = _FakeChatCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def _hit() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="content",
        chunk_index=0,
        page_number=1,
    )


def _patch_openai(
    monkeypatch: MonkeyPatch, content: str
) -> _FakeChatCompletions:
    fake = _FakeOpenAIClient(content)
    monkeypatch.setattr(
        "app.infrastructure.reranking.llm_reranker.OpenAI",
        lambda api_key: fake,
    )
    return fake.completions


def test_rerank_orders_by_llm_scores(monkeypatch: MonkeyPatch) -> None:
    hits = [_hit(), _hit(), _hit()]
    scores = [
        {"id": str(hits[1].chunk_id), "score": 9},
        {"id": str(hits[0].chunk_id), "score": 5},
        {"id": str(hits[2].chunk_id), "score": 2},
    ]
    _patch_openai(monkeypatch, json.dumps({"scores": scores}))

    reranker = LLMReranker(api_key="test-key")
    reranked = reranker.rerank("the query", hits)

    assert [hit.chunk_id for hit in reranked] == [
        hits[1].chunk_id,
        hits[0].chunk_id,
        hits[2].chunk_id,
    ]
    assert reranked[0].final_score == pytest.approx(9.0)
    assert [hit.rank for hit in reranked] == [1, 2, 3]


def test_rerank_treats_unscored_chunks_as_zero(monkeypatch: MonkeyPatch) -> None:
    scored = _hit()
    unscored = _hit()
    _patch_openai(
        monkeypatch,
        f'{{"scores": [{{"id": "{scored.chunk_id}", "score": 8}}]}}',
    )

    reranker = LLMReranker(api_key="test-key")
    reranked = reranker.rerank("the query", [unscored, scored])

    assert reranked[0].chunk_id == scored.chunk_id
    assert reranked[1].chunk_id == unscored.chunk_id
    assert reranked[1].final_score == pytest.approx(0.0)


def test_rerank_uses_json_object_response_format(monkeypatch: MonkeyPatch) -> None:
    completions = _patch_openai(monkeypatch, '{"scores": []}')

    reranker = LLMReranker(api_key="test-key")
    reranker.rerank("the query", [_hit()])

    kwargs = completions.calls[0]
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "the query" in kwargs["messages"][0]["content"]


def test_rerank_truncates_long_excerpts(monkeypatch: MonkeyPatch) -> None:
    completions = _patch_openai(monkeypatch, '{"scores": []}')
    long_hit = _hit()
    long_hit.content = "x" * 5000

    reranker = LLMReranker(
        api_key="test-key", max_excerpt_chars=100
    )
    reranker.rerank("the query", [long_hit])

    prompt = completions.calls[0]["messages"][0]["content"]
    assert "x" * 100 + "..." in prompt
    assert "x" * 5000 not in prompt


def test_rerank_empty_results_returns_empty_list(monkeypatch: MonkeyPatch) -> None:
    _patch_openai(monkeypatch, '{"scores": []}')
    reranker = LLMReranker(api_key="test-key")

    assert reranker.rerank("the query", []) == []


def test_reranker_rejects_missing_api_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        LLMReranker(api_key="")


def test_reranker_rejects_non_positive_excerpt_chars() -> None:
    with pytest.raises(ValueError, match="max_excerpt_chars"):
        LLMReranker(api_key="test-key", max_excerpt_chars=0)