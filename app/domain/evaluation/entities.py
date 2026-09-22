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


@dataclass(slots=True)
class AnswerEvaluation:
    """The outcome of evaluating a single generated answer."""

    query: str
    answer: str
    context: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_relevancy: float
    top_k: int


@dataclass(slots=True)
class GenerationReport:
    """Aggregate generation metrics over a set of golden queries."""

    num_queries: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_context_relevancy: float
    per_answer: list[AnswerEvaluation] = field(default_factory=list)