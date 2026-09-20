import pytest
from pytest import MonkeyPatch

from app.core.settings import Settings
from app.infrastructure.llm.canned_provider import CannedLLMProvider
from app.infrastructure.llm.factory import (
    UnknownLLMProviderError,
    build_llm_provider,
)
from app.infrastructure.llm.openai_provider import OpenAILLMProvider


class _FakeChatCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message", (), {"content": self.content}
                            )()
                        },
                    )()
                ]
            },
        )()


class _FakeOpenAIClient:
    def __init__(self, content: str) -> None:
        self.completions = _FakeChatCompletions(content)
        self.chat = type("Chat", (), {"completions": self.completions})()


def _patch_openai(
    monkeypatch: MonkeyPatch, content: str
) -> _FakeChatCompletions:
    fake = _FakeOpenAIClient(content)
    monkeypatch.setattr(
        "app.infrastructure.llm.openai_provider.OpenAI",
        lambda api_key: fake,
    )
    return fake.completions


def test_generate_sends_system_and_user_prompts(monkeypatch: MonkeyPatch) -> None:
    completions = _patch_openai(monkeypatch, "the completion")

    provider = OpenAILLMProvider(api_key="test-key", model="gpt-4o")
    result = provider.generate("be helpful", "answer this")

    assert result == "the completion"
    assert provider.model == "gpt-4o"
    assert completions.calls[0]["model"] == "gpt-4o"
    assert completions.calls[0]["messages"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "answer this"},
    ]


def test_generate_uses_per_call_max_tokens_override(
    monkeypatch: MonkeyPatch,
) -> None:
    completions = _patch_openai(monkeypatch, "x")

    provider = OpenAILLMProvider(api_key="test-key", max_tokens=256)
    provider.generate("sys", "user", max_tokens=512)

    assert completions.calls[0]["max_tokens"] == 512


def test_generate_falls_back_to_configured_max_tokens(
    monkeypatch: MonkeyPatch,
) -> None:
    completions = _patch_openai(monkeypatch, "x")

    provider = OpenAILLMProvider(api_key="test-key", max_tokens=256)
    provider.generate("sys", "user")

    assert completions.calls[0]["max_tokens"] == 256


def test_provider_rejects_missing_api_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        OpenAILLMProvider(api_key="")


def test_provider_rejects_non_positive_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        OpenAILLMProvider(api_key="test-key", max_tokens=0)


def test_canned_provider_cites_every_source() -> None:
    provider = CannedLLMProvider()
    prompt = (
        "Question: how does it work?\n\nDocumentation:\n"
        "[1] Guide, page 1\nalpha\n\n[2] Manual, page 3\nbeta"
    )

    answer = provider.generate("system", prompt)

    assert answer == (
        "Based on the documentation provided, the answer to this question is "
        "supported by the following sources: [1], [2]."
    )
    assert provider.model == "devrag-canned"


def test_canned_provider_returns_empty_answer_without_sources() -> None:
    provider = CannedLLMProvider()

    answer = provider.generate("system", "Question: how does it work?")

    assert "does not contain enough information" in answer


def test_factory_builds_openai_provider(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    app_settings = Settings(
        generation_provider="openai",
        generation_model="gpt-4o",
        generation_max_tokens=1024,
    )

    provider = build_llm_provider(app_settings)

    assert isinstance(provider, OpenAILLMProvider)
    assert provider._model == "gpt-4o"
    assert provider._max_tokens == 1024


def test_factory_builds_canned_provider() -> None:
    app_settings = Settings(generation_provider="none")

    provider = build_llm_provider(app_settings)

    assert isinstance(provider, CannedLLMProvider)


def test_factory_raises_for_unknown_provider() -> None:
    app_settings = Settings(generation_provider="anthropic")

    with pytest.raises(UnknownLLMProviderError, match="anthropic"):
        build_llm_provider(app_settings)