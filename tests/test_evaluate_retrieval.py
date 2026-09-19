from uuid import uuid4

import pytest

from app.application.evaluation.evaluate_retrieval import EvaluateRetrieval
from app.domain.evaluation.entities import GoldenQuery
from app.domain.retrieval.entities import RetrievedChunk


class _FakeLoader:
    def __init__(self, queries: list[GoldenQuery]) -> None:
        self.queries = queries
        self.calls = 0

    def load(self) -> list[GoldenQuery]:
        self.calls += 1
        return self.queries


class _FakeRetriever:
    def __init__(self, results: dict[str, list[RetrievedChunk]]) -> None:
        self.results = results

    def __call__(self, query: str) -> list[RetrievedChunk]:
        return self.results.get(query, [])


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="content",
        chunk_index=0,
        page_number=1,
    )


def _golden(query: str, relevant: list[RetrievedChunk]) -> GoldenQuery:
    return GoldenQuery(
        query=query,
        relevant_chunk_ids=[chunk.chunk_id for chunk in relevant],
    )


def test_execute_aggregates_hit_rate_and_mrr() -> None:
    top = _chunk()
    second = _chunk()
    missed = _chunk()
    unrelated = _chunk()
    queries = [
        _golden("top hit", [top]),
        _golden("second hit", [second]),
        _golden("missed", [missed]),
    ]
    retriever = _FakeRetriever(
        {
            "top hit": [top, unrelated],
            "second hit": [unrelated, second, top],
            "missed": [unrelated],
        }
    )
    use_case = EvaluateRetrieval(_FakeLoader(queries), retriever, top_k=5)

    report = use_case.execute()

    assert report.num_queries == 3
    assert report.hit_rate == pytest.approx(2 / 3)
    assert report.mean_reciprocal_rank == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_execute_truncates_results_to_top_k() -> None:
    # Relevant chunk sits at position 6 -> miss under top_k=5, hit under top_k=10.
    far_relevant = _chunk()
    rank_6_results = [_chunk() for _ in range(5)] + [far_relevant]

    strict = EvaluateRetrieval(
        _FakeLoader([_golden("deep hit", [far_relevant])]),
        _FakeRetriever({"deep hit": rank_6_results}),
        top_k=5,
    )
    loose = EvaluateRetrieval(
        _FakeLoader([_golden("deep hit", [far_relevant])]),
        _FakeRetriever({"deep hit": rank_6_results}),
        top_k=10,
    )

    assert strict.execute().hit_rate == pytest.approx(0.0)
    assert loose.execute().hit_rate == pytest.approx(1.0)


def test_execute_uses_retrieved_chunk_ids_in_per_query_details() -> None:
    hit = _chunk()
    other = _chunk()
    use_case = EvaluateRetrieval(
        _FakeLoader([_golden("query", [hit])]),
        _FakeRetriever({"query": [hit, other]}),
        top_k=5,
    )

    report = use_case.execute()

    (evaluation,) = report.per_query
    assert evaluation.query == "query"
    assert evaluation.top_k == 5
    assert evaluation.retrieved_chunk_ids == [hit.chunk_id, other.chunk_id]
    assert evaluation.reciprocal_rank == pytest.approx(1.0)


def test_execute_loads_queries_from_the_loader() -> None:
    top = _chunk()
    loader = _FakeLoader([_golden("query", [top])])
    use_case = EvaluateRetrieval(loader, _FakeRetriever({"query": [top]}))

    use_case.execute()

    assert loader.calls == 1


def test_execute_raises_for_empty_golden_set() -> None:
    use_case = EvaluateRetrieval(
        _FakeLoader([]), _FakeRetriever({}), top_k=5
    )

    with pytest.raises(ValueError, match="not be empty"):
        use_case.execute()


def test_execute_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        EvaluateRetrieval(_FakeLoader([]), _FakeRetriever({}), top_k=0)