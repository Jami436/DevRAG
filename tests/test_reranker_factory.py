import pytest

from app.core.settings import Settings
from app.infrastructure.reranking.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from app.infrastructure.reranking.factory import (
    UnknownRerankerProviderError,
    build_reranker,
)
from app.infrastructure.reranking.identity_reranker import IdentityReranker
from app.infrastructure.reranking.llm_reranker import LLMReranker


def test_factory_builds_cross_encoder_reranker() -> None:
    settings = Settings(reranker_provider="cross_encoder")

    reranker = build_reranker(settings)

    assert isinstance(reranker, CrossEncoderReranker)


def test_factory_builds_llm_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = Settings(reranker_provider="llm")

    reranker = build_reranker(settings)

    assert isinstance(reranker, LLMReranker)


def test_factory_builds_identity_reranker() -> None:
    settings = Settings(reranker_provider="none")

    reranker = build_reranker(settings)

    assert isinstance(reranker, IdentityReranker)


def test_factory_raises_for_unknown_provider() -> None:
    settings = Settings(reranker_provider="bm25")

    with pytest.raises(UnknownRerankerProviderError, match="bm25"):
        build_reranker(settings)


def test_factory_uses_config_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = Settings(
        reranker_provider="llm",
        llm_reranker_model="gpt-4o",
    )

    reranker = build_reranker(settings)

    assert isinstance(reranker, LLMReranker)
    assert reranker._model == "gpt-4o"


def test_cross_encoder_uses_configured_model() -> None:
    settings = Settings(
        reranker_provider="cross_encoder",
        reranker_model="cross-encoder/custom-model",
    )

    reranker = build_reranker(settings)

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker._model_name == "cross-encoder/custom-model"