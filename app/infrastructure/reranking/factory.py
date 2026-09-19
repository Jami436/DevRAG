import os

from app.core.settings import Settings
from app.domain.retrieval.interfaces import Reranker
from app.infrastructure.reranking.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from app.infrastructure.reranking.identity_reranker import IdentityReranker
from app.infrastructure.reranking.llm_reranker import LLMReranker


class UnknownRerankerProviderError(ValueError):
    """Raised when no reranker is available for the configured provider."""


def build_reranker(settings: Settings) -> Reranker:
    """Build the reranker configured by ``settings``.

    Supports ``cross_encoder`` (on-device sentence-transformers model),
    ``llm`` (OpenAI chat-based scoring) and ``none`` (no-op). The OpenAI API
    key falls back to the ``OPENAI_API_KEY`` environment variable when not
    present in settings.
    """
    provider = settings.reranker_provider
    if provider == "cross_encoder":
        return CrossEncoderReranker(model_name=settings.reranker_model)
    if provider == "llm":
        api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        return LLMReranker(api_key=api_key, model=settings.llm_reranker_model)
    if provider == "none":
        return IdentityReranker()
    raise UnknownRerankerProviderError(
        f"Unknown reranker provider: {provider}"
    )