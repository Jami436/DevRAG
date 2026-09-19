from typing import ClassVar
from uuid import uuid4

import numpy as np
import pytest

from app.domain.retrieval.entities import RetrievedChunk
from app.infrastructure.reranking.cross_encoder_reranker import (
    CrossEncoderReranker,
)


class _FakeCrossEncoder:
    """Stub of sentence_transformers.CrossEncoder for offline tests."""

    def __init__(
        self, model_name: str, scores: list[float] | None = None
    ) -> None:
        self.model_name = model_name
        self.scores = scores if scores is not None else [1.0, 5.0, 3.0]
        self.recorded: list[tuple[str, str]] = []

    def predict(self, sentences: list[tuple[str, str]]) -> np.ndarray:
        self.recorded.extend(sentences)
        return np.array(self.scores[: len(sentences)], dtype=float)


def _hit(content: str = "content") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content=content,
        chunk_index=0,
        page_number=1,
    )


def test_rerank_orders_by_descending_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.reranking.cross_encoder_reranker.CrossEncoder",
        _FakeCrossEncoder,
    )
    reranker = CrossEncoderReranker(model_name="fake-model")
    hits = [_hit(), _hit(), _hit()]

    reranked = reranker.rerank("query", hits)

    assert [hit.rank for hit in reranked] == [1, 2, 3]
    assert reranked[0].final_score == pytest.approx(5.0)
    assert reranked[-1].final_score == pytest.approx(1.0)


def test_rerank_preserves_chunks_on_ties(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TiedEncoder(_FakeCrossEncoder):
        def __init__(self, model_name: str) -> None:
            super().__init__(model_name, scores=[2.0, 2.0, 2.0])

    monkeypatch.setattr(
        "app.infrastructure.reranking.cross_encoder_reranker.CrossEncoder",
        _TiedEncoder,
    )
    reranker = CrossEncoderReranker(model_name="fake-model")
    hits = [_hit(), _hit(), _hit()]

    reranked = reranker.rerank("query", hits)

    assert [hit.rank for hit in reranked] == [1, 2, 3]
    assert len(reranked) == 3


def test_rerank_scores_each_pair_with_query_and_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCrossEncoder("fake-model", scores=[3.0, 1.0])

    monkeypatch.setattr(
        "app.infrastructure.reranking.cross_encoder_reranker.CrossEncoder",
        lambda model_name: fake,
    )
    reranker = CrossEncoderReranker(model_name="fake-model")
    hits = [_hit("alpha"), _hit("beta")]

    reranker.rerank("the query", hits)

    assert fake.recorded == [("the query", "alpha"), ("the query", "beta")]


def test_rerank_empty_results_returns_empty_list() -> None:
    reranker = CrossEncoderReranker(model_name="fake-model")

    assert reranker.rerank("query", []) == []


def test_rerank_model_loads_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RecordingEncoder(_FakeCrossEncoder):
        instances: ClassVar[list[str]] = []

        def __init__(self, model_name: str) -> None:
            super().__init__(model_name)
            _RecordingEncoder.instances.append(model_name)

    monkeypatch.setattr(
        "app.infrastructure.reranking.cross_encoder_reranker.CrossEncoder",
        _RecordingEncoder,
    )
    reranker = CrossEncoderReranker(model_name="fake-model")
    assert _RecordingEncoder.instances == []

    reranker.rerank("query", [_hit()])

    assert _RecordingEncoder.instances == ["fake-model"]


def test_reranker_rejects_empty_model_name() -> None:
    with pytest.raises(ValueError, match="model_name"):
        CrossEncoderReranker(model_name="")