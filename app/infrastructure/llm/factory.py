import os

from app.core.settings import Settings
from app.domain.generation.interfaces import LLMProvider
from app.infrastructure.llm.canned_provider import CannedLLMProvider
from app.infrastructure.llm.openai_provider import OpenAILLMProvider


class UnknownLLMProviderError(ValueError):
    """Raised when no adapter is available for the configured provider."""


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Build the LLM provider configured by ``settings``.

    Supports ``openai`` (remote Chat Completions API) and ``none`` (offline
    canned placeholder). The OpenAI API key falls back to the
    ``OPENAI_API_KEY`` environment variable when not present in settings.
    """
    provider = settings.generation_provider
    if provider == "openai":
        api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        return OpenAILLMProvider(
            api_key=api_key,
            model=settings.generation_model,
            max_tokens=settings.generation_max_tokens,
        )
    if provider == "none":
        return CannedLLMProvider()
    raise UnknownLLMProviderError(f"Unknown LLM provider: {provider}")