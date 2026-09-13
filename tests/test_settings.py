import pytest

from app.core.settings import Settings


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_NAME", raising=False)
    monkeypatch.delenv("APP_VERSION", raising=False)
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.delenv("API_V1_PREFIX", raising=False)

    settings = Settings(_env_file=None)

    assert settings.app_name == "DevRAG"
    assert settings.app_version == "0.1.0"
    assert settings.debug is True
    assert settings.api_v1_prefix == "/api/v1"


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_NAME", "TestRAG")
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("API_V1_PREFIX", "/v2")

    settings = Settings(_env_file=None)

    assert settings.app_name == "TestRAG"
    assert settings.app_version == "9.9.9"
    assert settings.debug is False
    assert settings.api_v1_prefix == "/v2"
