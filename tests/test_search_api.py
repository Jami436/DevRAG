from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.v1.search import get_search_service
from app.domain.retrieval.entities import RetrievedChunk
from app.main import app


class _FakeSearchService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def execute(
        self, query: str, *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return [
            RetrievedChunk(
                chunk_id=uuid4(),
                document_id=uuid4(),
                content="retrieved chunk content",
                chunk_index=0,
                page_number=1,
                metadata={"heading": "Intro"},
                document_title="Guide",
                vector_score=0.92,
                keyword_score=0.4,
                final_score=0.75,
                rank=1,
            )
        ]


@pytest.fixture()
def search_endpoint() -> Iterator[tuple[TestClient, _FakeSearchService]]:
    fake = _FakeSearchService()
    app.dependency_overrides[get_search_service] = lambda: fake
    with TestClient(app) as client:
        yield client, fake
    app.dependency_overrides.clear()


def _post(client: TestClient, payload: dict[str, object]) -> Any:
    return client.post("/api/v1/search", json=payload)


def test_search_endpoint_returns_reranked_hits(
    search_endpoint: tuple[TestClient, _FakeSearchService],
) -> None:
    client, _fake = search_endpoint

    response = _post(client, {"query": "vector databases", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "vector databases"
    (result,) = body["results"]
    assert result["content"] == "retrieved chunk content"
    assert result["document_title"] == "Guide"
    assert result["vector_score"] == 0.92
    assert result["keyword_score"] == 0.4
    assert result["final_score"] == 0.75
    assert result["rank"] == 1
    assert result["metadata"] == {"heading": "Intro"}


def test_search_endpoint_forwards_request_top_k(
    search_endpoint: tuple[TestClient, _FakeSearchService],
) -> None:
    client, fake = search_endpoint

    _post(client, {"query": "question", "top_k": 2})

    assert fake.calls == [("question", 2)]


def test_search_endpoint_uses_settings_default_top_k(
    search_endpoint: tuple[TestClient, _FakeSearchService],
) -> None:
    client, fake = search_endpoint

    _post(client, {"query": "question"})

    assert fake.calls == [("question", 5)]


def test_search_endpoint_rejects_empty_query(
    search_endpoint: tuple[TestClient, _FakeSearchService],
) -> None:
    client, _fake = search_endpoint

    assert _post(client, {"query": ""}).status_code == 422


def test_search_endpoint_rejects_out_of_range_top_k(
    search_endpoint: tuple[TestClient, _FakeSearchService],
) -> None:
    client, _fake = search_endpoint

    assert _post(client, {"query": "q", "top_k": 0}).status_code == 422
    assert _post(client, {"query": "q", "top_k": 21}).status_code == 422