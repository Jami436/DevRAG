import os

from app.core.settings import Settings
from app.domain.embeddings.interfaces import EmbeddingProvider
from app.infrastructure.embeddings.local_provider import LocalEmbeddingProvider
from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider


class UnknownEmbeddingProviderError(ValueError):
    """Raised when no adapter is available for the configured provider."""


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the embedding provider configured by ``settings``.

    Supports ``openai`` (remote API) and ``local`` (Hugging Face on-device).
    The OpenAI API key falls back to the ``OPENAI_API_KEY`` environment
    variable when not present in settings.
    """
    provider = settings.embedding_provider
    if provider == "openai":
        api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=settings.openai_embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    if provider == "local":
        return LocalEmbeddingProvider(
            model_name=settings.local_embedding_model,
        )
    raise UnknownEmbeddingProviderError(f"Unknown embedding provider: {provider}")