from dataclasses import dataclass, field
from uuid import UUID


@dataclass(slots=True)
class GoldenQuery:
    """A query paired with the chunk ids considered relevant (ground truth)."""

    query: str
    relevant_chunk_ids: list[UUID] = field(default_factory=list)


@dataclass(slots=True)
class QueryEvaluation:
    """The outcome of retrieving for a single golden query."""

    query: str
    relevant_chunk_ids: list[UUID]
    retrieved_chunk_ids: list[UUID]
    top_k: int
    hit: bool
    reciprocal_rank: float


@dataclass(slots=True)
class RetrievalReport:
    """Aggregate retrieval metrics over a set of golden queries."""

    num_queries: int
    hit_rate: float
    mean_reciprocal_rank: float
    per_query: list[QueryEvaluation] = field(default_factory=list)