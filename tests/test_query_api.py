from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.v1.query import get_query_service
from app.domain.generation.entities import Citation, GeneratedAnswer
from app.main import app

_CHUNK_ID = uuid4()
_DOCUMENT_ID = uuid4()


class _FakeQueryService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def execute(
        self, query: str, *, top_k: int | None = None
    ) -> GeneratedAnswer:
        self.calls.append((query, top_k))
        return GeneratedAnswer(
            query=query,
            answer=f"Answer about: {query}",
            citations=[
                Citation(
                    chunk_id=_CHUNK_ID,
                    document_id=_DOCUMENT_ID,
                    chunk_index=0,
                    page_number=3,
                    document_title="Guide",
                    excerpt="excerpt from the guide",
                    metadata={"heading": "Intro"},
                )
            ],
            model="pytest-llm",
        )


@pytest.fixture()
def query_endpoint() -> Iterator[tuple[TestClient, _FakeQueryService]]:
    fake = _FakeQueryService()
    app.dependency_overrides[get_query_service] = lambda: fake
    with TestClient(app) as client:
        yield client, fake
    app.dependency_overrides.clear()


def _post(client: TestClient, payload: dict[str, object]) -> Any:
    return client.post("/api/v1/query", json=payload)


def test_query_endpoint_returns_answer_with_citations(
    query_endpoint: tuple[TestClient, _FakeQueryService],
) -> None:
    client, _fake = query_endpoint

    response = _post(client, {"query": "how do I ingest docs?", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "how do I ingest docs?"
    assert body["answer"] == "Answer about: how do I ingest docs?"
    assert body["model"] == "pytest-llm"
    (citation,) = body["citations"]
    assert citation["chunk_id"] == str(_CHUNK_ID)
    assert citation["document_id"] == str(_DOCUMENT_ID)
    assert citation["document_title"] == "Guide"
    assert citation["page_number"] == 3
    assert citation["excerpt"] == "excerpt from the guide"
    assert citation["metadata"] == {"heading": "Intro"}


def test_query_endpoint_forwards_request_top_k(
    query_endpoint: tuple[TestClient, _FakeQueryService],
) -> None:
    client, fake = query_endpoint

    _post(client, {"query": "question", "top_k": 3})

    assert fake.calls == [("question", 3)]


def test_query_endpoint_calls_the_service(
    query_endpoint: tuple[TestClient, _FakeQueryService],
) -> None:
    client, fake = query_endpoint

    response = _post(client, {"query": "question"})

    assert response.status_code == 200
    assert fake.calls == [("question", 5)]


def test_query_endpoint_rejects_empty_query(
    query_endpoint: tuple[TestClient, _FakeQueryService],
) -> None:
    client, _fake = query_endpoint

    response = _post(client, {"query": ""})

    assert response.status_code == 422


def test_query_endpoint_rejects_out_of_range_top_k(
    query_endpoint: tuple[TestClient, _FakeQueryService],
) -> None:
    client, _fake = query_endpoint

    assert _post(client, {"query": "q", "top_k": 0}).status_code == 422
    assert _post(client, {"query": "q", "top_k": 21}).status_code == 422