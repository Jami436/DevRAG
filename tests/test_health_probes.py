from typing import cast

import pytest
from sqlalchemy import Engine, create_engine

from app.core.settings import Settings
from app.domain.health.entities import DependencyStatus
from app.infrastructure.health.api_key_probe import ApiKeyProbe
from app.infrastructure.health.database_probe import DatabaseProbe
from app.infrastructure.health.factory import build_default_health_probes


def test_database_probe_reports_ok_on_live_engine() -> None:
    status = DatabaseProbe(create_engine("sqlite://")).check()

    assert status == DependencyStatus(name="database", state="ok")


class _BoomEngine:
    def connect(self) -> None:
        raise OSError("connection refused")


def test_database_probe_reports_error_when_connection_fails() -> None:
    probe = DatabaseProbe(cast(Engine, _BoomEngine()))

    status = probe.check()

    assert status.name == "database"
    assert status.state == "error"
    assert "connection refused" in (status.detail or "")


def test_api_key_probe_reports_ok_when_key_configured() -> None:
    status = ApiKeyProbe(name="openai", api_key="sk-test").check()

    assert status == DependencyStatus(name="openai", state="ok")


def test_api_key_probe_reports_error_when_key_missing() -> None:
    status = ApiKeyProbe(name="openai", api_key=None).check()

    assert status.state == "error"
    assert status.detail == "API key is not configured"


def test_factory_builds_database_and_openai_probes_by_default() -> None:
    probes = build_default_health_probes(Settings(openai_api_key="sk-test"))

    assert isinstance(probes[0], DatabaseProbe)
    assert isinstance(probes[1], ApiKeyProbe)
    assert probes[1].check() == DependencyStatus(name="openai", state="ok")


def test_factory_skips_openai_probe_when_no_openai_provider_used() -> None:
    app_settings = Settings(
        embedding_provider="local",
        reranker_provider="cross_encoder",
        generation_provider="none",
    )

    probes = build_default_health_probes(app_settings)

    assert [type(probe).__name__ for probe in probes] == ["DatabaseProbe"]


@pytest.mark.parametrize(
    ("embedding_provider", "reranker_provider", "generation_provider"),
    [
        ("openai", "cross_encoder", "none"),
        ("local", "llm", "none"),
        ("local", "cross_encoder", "openai"),
    ],
)
def test_factory_probes_openai_when_any_provider_uses_it(
    embedding_provider: str,
    reranker_provider: str,
    generation_provider: str,
) -> None:
    app_settings = Settings(
        embedding_provider=embedding_provider,
        reranker_provider=reranker_provider,
        generation_provider=generation_provider,
        openai_api_key="sk-test",
    )

    probes = build_default_health_probes(app_settings)

    assert [type(probe).__name__ for probe in probes] == [
        "DatabaseProbe",
        "ApiKeyProbe",
    ]
    assert probes[1].check() == DependencyStatus(name="openai", state="ok")