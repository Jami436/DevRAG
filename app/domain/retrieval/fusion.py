from collections.abc import Sequence
from uuid import UUID

from app.domain.retrieval.entities import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    limit: int,
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse ranked result lists with Reciprocal Rank Fusion (RRF).

    Every list contributes ``weight / (k + rank)`` to a chunk's fused score,
    where ``rank`` is the 1-based position within that list. Fusing ranks
    instead of raw scores sidesteps the fact that different signals live on
    incommensurable scales (cosine distance versus full-text rank). The merged
    chunks are sorted by descending fused score, truncated to ``limit`` and
    tagged with their fused score and final rank.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    if k <= 0:
        raise ValueError("k must be a positive constant")
    if not ranked_lists:
        return []
    effective_weights = (
        [1.0] * len(ranked_lists) if weights is None else list(weights)
    )
    if len(effective_weights) != len(ranked_lists):
        raise ValueError("weights must align one-to-one with ranked lists")

    by_chunk: dict[UUID, RetrievedChunk] = {}
    fused_scores: dict[UUID, float] = {}
    for contributions, weight in zip(
        ranked_lists, effective_weights, strict=True
    ):
        for rank, hit in enumerate(contributions, start=1):
            by_chunk[hit.chunk_id] = hit
            increment = weight / (k + rank)
            fused_scores[hit.chunk_id] = (
                fused_scores.get(hit.chunk_id, 0.0) + increment
            )

    ordered = sorted(
        fused_scores.items(), key=lambda item: item[1], reverse=True
    )[:limit]
    fused: list[RetrievedChunk] = []
    for rank, (chunk_id, score) in enumerate(ordered, start=1):
        hit = by_chunk[chunk_id]
        hit.final_score = score
        hit.rank = rank
        fused.append(hit)
    return fused