import os

from app.core.config import settings
from app.core.settings import Settings
from app.domain.health.interfaces import HealthProbe
from app.infrastructure.db.session import engine
from app.infrastructure.health.api_key_probe import ApiKeyProbe
from app.infrastructure.health.database_probe import DatabaseProbe


def build_default_health_probes(
    app_settings: Settings = settings,
) -> list[HealthProbe]:
    """Wire the default dependency probes from application settings.

    Database connectivity is always probed; OpenAI-related providers are probed
    only when a provider that depends on them is configured.
    """
    probes: list[HealthProbe] = [DatabaseProbe(engine)]

    needs_openai = (
        app_settings.embedding_provider == "openai"
        or app_settings.reranker_provider == "llm"
        or app_settings.generation_provider == "openai"
    )
    if needs_openai:
        api_key = app_settings.openai_api_key or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        probes.append(ApiKeyProbe(name="openai", api_key=api_key or None))

    return probes