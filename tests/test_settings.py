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
    assert settings.log_level == "INFO"
    assert settings.log_format == "json"
    assert settings.retriever_top_k == 5
    assert settings.retrieval_hybrid_candidates == 50
    assert settings.retrieval_fusion_k == 60
    assert settings.reranker_provider == "cross_encoder"
    assert settings.generation_provider == "openai"
    assert settings.generation_model == "gpt-4o-mini"
    assert settings.generation_max_tokens == 800
    assert settings.generation_max_excerpt_chars == 1000
    assert settings.generation_max_context_chars == 12000
    assert (
        settings.evaluation_golden_queries_path
        == "data/evaluation/golden_queries.json"
    )


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_NAME", "TestRAG")
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("API_V1_PREFIX", "/v2")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "text")

    settings = Settings(_env_file=None)

    assert settings.app_name == "TestRAG"
    assert settings.app_version == "9.9.9"
    assert settings.debug is False
    assert settings.api_v1_prefix == "/v2"
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "text"
