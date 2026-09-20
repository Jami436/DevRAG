from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.v1.health import get_health_probes, health_check
from app.core.config import settings
from app.domain.health.entities import DependencyStatus
from app.main import app


def test_health_check_returns_status() -> None:
    response = health_check()

    assert response == {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


def test_liveness_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


def test_version_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/version")

    assert response.status_code == 200
    assert response.json() == {
        "service": settings.app_name,
        "version": settings.app_version,
    }


class _FakeProbe:
    def __init__(self, status: DependencyStatus) -> None:
        self._status = status

    def check(self) -> DependencyStatus:
        return self._status


@pytest.fixture()
def clear_probe_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
def ready_probes(clear_probe_overrides: None) -> Iterator[None]:
    app.dependency_overrides[get_health_probes] = lambda: [
        _FakeProbe(DependencyStatus(name="database", state="ok")),
        _FakeProbe(DependencyStatus(name="openai", state="ok")),
    ]
    yield


@pytest.fixture()
def failing_probes(clear_probe_overrides: None) -> Iterator[None]:
    app.dependency_overrides[get_health_probes] = lambda: [
        _FakeProbe(
            DependencyStatus(
                name="database", state="error", detail="connection refused"
            )
        ),
    ]
    yield


def test_readiness_endpoint_reports_ready_when_all_probes_pass(
    ready_probes: None,
) -> None:
    client = TestClient(app)

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["service"] == settings.app_name
    assert body["version"] == settings.app_version
    assert body["dependencies"] == [
        {"name": "database", "state": "ok", "detail": None},
        {"name": "openai", "state": "ok", "detail": None},
    ]


def test_readiness_endpoint_reports_unhealthy_on_failed_probe(
    failing_probes: None,
) -> None:
    client = TestClient(app)

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["dependencies"] == [
        {
            "name": "database",
            "state": "error",
            "detail": "connection refused",
        },
    ]