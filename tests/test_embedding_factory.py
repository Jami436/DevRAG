import pytest

from app.core.settings import Settings
from app.infrastructure.embeddings.factory import (
    UnknownEmbeddingProviderError,
    build_embedding_provider,
)
from app.infrastructure.embeddings.local_provider import LocalEmbeddingProvider
from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider


def test_factory_builds_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = Settings(embedding_provider="openai")

    provider = build_embedding_provider(settings)

    assert isinstance(provider, OpenAIEmbeddingProvider)


def test_factory_builds_local_provider() -> None:
    settings = Settings(embedding_provider="local")

    provider = build_embedding_provider(settings)

    assert isinstance(provider, LocalEmbeddingProvider)


def test_factory_raises_for_unknown_provider() -> None:
    settings = Settings(embedding_provider="voyage")

    with pytest.raises(UnknownEmbeddingProviderError, match="voyage"):
        build_embedding_provider(settings)


def test_factory_uses_config_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = Settings(
        embedding_provider="openai",
        openai_embedding_model="text-embedding-3-large",
        embedding_batch_size=512,
    )

    provider = build_embedding_provider(settings)

    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider._model == "text-embedding-3-large"
    assert provider._batch_size == 512