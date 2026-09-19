from collections.abc import Collection, Sequence
from uuid import UUID


def is_hit(
    retrieved_chunk_ids: Sequence[UUID],
    relevant_chunk_ids: Collection[UUID],
) -> bool:
    """Return True when any retrieved chunk is in the relevant set."""
    return any(chunk_id in relevant_chunk_ids for chunk_id in retrieved_chunk_ids)


def reciprocal_rank(
    retrieved_chunk_ids: Sequence[UUID],
    relevant_chunk_ids: Collection[UUID],
) -> float:
    """Return the reciprocal of the first relevant chunk's rank, or 0.

    Ranks are 1-based. A chunk at position ``n`` contributes ``1 / n``; when no
    retrieved chunk is relevant the contribution is zero.
    """
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0