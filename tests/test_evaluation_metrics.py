from uuid import UUID, uuid4

import pytest

from app.domain.evaluation.metrics import is_hit, reciprocal_rank


def _ids(count: int) -> list[UUID]:
    return [uuid4() for _ in range(count)]


def test_is_hit_true_when_relevant_is_retrieved() -> None:
    retrieved = _ids(3)
    relevant = [retrieved[1]]

    assert is_hit(retrieved, relevant) is True


def test_is_hit_true_when_any_relevant_is_retrieved() -> None:
    retrieved = _ids(3)
    relevant = [uuid4(), retrieved[2]]

    assert is_hit(retrieved, relevant) is True


def test_is_hit_false_when_no_relevant_is_retrieved() -> None:
    retrieved = _ids(3)
    relevant = [uuid4()]

    assert is_hit(retrieved, relevant) is False


def test_is_hit_false_on_empty_retrieved() -> None:
    assert is_hit([], [uuid4()]) is False


def test_reciprocal_rank_is_one_for_top_result() -> None:
    retrieved = _ids(3)

    assert reciprocal_rank(retrieved, [retrieved[0]]) == pytest.approx(1.0)


def test_reciprocal_rank_uses_position_of_first_match() -> None:
    retrieved = _ids(3)
    relevant = [uuid4(), retrieved[2]]

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_unmatched() -> None:
    retrieved = _ids(3)

    assert reciprocal_rank(retrieved, [uuid4()]) == pytest.approx(0.0)


def test_reciprocal_rank_is_zero_on_empty_retrieved() -> None:
    assert reciprocal_rank([], [uuid4()]) == pytest.approx(0.0)