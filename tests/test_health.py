import asyncio

from fastapi.testclient import TestClient

from app.api.v1.health import health_check
from app.main import app


def test_health_check_returns_status() -> None:
    response = asyncio.run(health_check())

    assert response == {"status": "healthy", "service": "DevRAG"}


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "DevRAG"}
