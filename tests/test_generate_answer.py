from uuid import uuid4

import pytest

from app.application.generation.generate_answer import GenerateAnswer
from app.domain.retrieval.entities import RetrievedChunk


class _FakeLLMProvider:
    def __init__(self, text: str = "The answer is [2].") -> None:
        self.text = text
        self.calls: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        del max_tokens
        self.calls.append((system_prompt, user_prompt))
        return self.text


def _hit(content: str = "content", title: str = "Guide") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content=content,
        chunk_index=0,
        page_number=2,
        document_title=title,
    )


def test_execute_generates_and_cites() -> None:
    second = _hit(content="beta")
    first = _hit(content="alpha")
    provider = _FakeLLMProvider("The beta [2] and alpha [1] sources.")

    answer = GenerateAnswer(provider).execute("the query", [first, second])

    assert answer.query == "the query"
    assert answer.answer == "The beta [2] and alpha [1] sources."
    assert answer.model == "fake-model"
    assert [citation.chunk_id for citation in answer.citations] == [
        second.chunk_id,
        first.chunk_id,
    ]
    assert [citation.excerpt for citation in answer.citations] == ["beta", "alpha"]


def test_execute_sends_grounding_system_and_numbered_user_prompts() -> None:
    chunk = _hit(content="alpha")
    provider = _FakeLLMProvider()
    GenerateAnswer(provider).execute("the query", [chunk])

    system_prompt, user_prompt = provider.calls[0]
    assert system_prompt.startswith("You are an expert assistant")
    assert user_prompt.startswith("Question: the query")
    assert "[1] Guide, page 2\nalpha" in user_prompt


def test_execute_with_empty_chunks_does_not_cite() -> None:
    provider = _FakeLLMProvider("I cannot answer this question.")

    answer = GenerateAnswer(provider).execute("the query", [])

    assert answer.citations == []
    assert "No relevant documentation was found" in provider.calls[0][1]


def test_execute_configures_context_budget() -> None:
    long_chunk = _hit(content="x" * 500)
    provider = _FakeLLMProvider()
    GenerateAnswer(provider, max_excerpt_chars=100).execute(
        "the query", [long_chunk]
    )

    _system_prompt, user_prompt = provider.calls[0]
    assert ("x" * 100) + "..." in user_prompt
    assert "x" * 500 not in user_prompt


def test_execute_rejects_non_positive_budgets() -> None:
    with pytest.raises(ValueError, match="max_excerpt_chars"):
        GenerateAnswer(_FakeLLMProvider(), max_excerpt_chars=0)
    with pytest.raises(ValueError, match="max_total_chars"):
        GenerateAnswer(_FakeLLMProvider(), max_total_chars=0)