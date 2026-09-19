from uuid import uuid4

import pytest

from app.domain.retrieval.entities import RetrievedChunk
from app.domain.retrieval.fusion import reciprocal_rank_fusion


def _hit() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="content",
        chunk_index=0,
        page_number=1,
    )


def test_chunks_present_in_multiple_lists_rank_first() -> None:
    shared = _hit()
    listed = _hit()
    only = _hit()

    fused = reciprocal_rank_fusion(
        ([shared, listed], [shared, only]),
        limit=3,
    )

    assert [hit.chunk_id for hit in fused] == [
        shared.chunk_id,
        listed.chunk_id,
        only.chunk_id,
    ]
    assert fused[0].final_score == pytest.approx(1 / 61 + 1 / 61)
    assert fused[1].final_score == pytest.approx(1 / 62)


def test_fused_scores_decrease_with_rank() -> None:
    first = _hit()
    second = _hit()
    third = _hit()

    fused = reciprocal_rank_fusion(
        ([first, second, third],),
        limit=3,
    )

    assert [hit.rank for hit in fused] == [1, 2, 3]
    scores = [hit.final_score or 0.0 for hit in fused]
    assert scores[0] > scores[1] > scores[2]


def test_weights_favour_one_signal() -> None:
    vector_win = _hit()
    keyword_win = _hit()
    vector_only = _hit()
    keyword_only = _hit()

    fused = reciprocal_rank_fusion(
        ([vector_win, vector_only], [keyword_only, keyword_win]),
        weights=(3.0, 1.0),
        limit=4,
    )

    # Heavy vector weight pushes the vector-only hit above the keyword-only one.
    assert [hit.chunk_id for hit in fused] == [
        vector_win.chunk_id,
        vector_only.chunk_id,
        keyword_only.chunk_id,
        keyword_win.chunk_id,
    ]


def test_limit_truncates_fused_results() -> None:
    hits = [_hit() for _ in range(5)]

    fused = reciprocal_rank_fusion((hits,), limit=2)

    assert len(fused) == 2
    assert fused[0].rank == 1
    assert fused[1].rank == 2


def test_empty_lists_yield_empty_result() -> None:
    assert reciprocal_rank_fusion((), limit=5) == []
    assert reciprocal_rank_fusion(([], []), limit=5) == []


def test_weights_must_align_with_lists() -> None:
    with pytest.raises(ValueError, match="align"):
        reciprocal_rank_fusion(([_hit()], [_hit()]), weights=(1.0,), limit=2)


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="limit"):
        reciprocal_rank_fusion(([_hit()],), limit=0)


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion(([_hit()],), limit=1, k=0)