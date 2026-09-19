from collections.abc import Callable

from app.domain.evaluation.entities import (
    GoldenQuery,
    QueryEvaluation,
    RetrievalReport,
)
from app.domain.evaluation.interfaces import GoldenQueryLoader
from app.domain.evaluation.metrics import is_hit, reciprocal_rank
from app.domain.retrieval.entities import RetrievedChunk


class EvaluateRetrieval:
    """Score a retriever's hit rate and MRR against golden queries.

    Each golden query is run through the injected retriever, results are
    truncated to ``top_k`` and compared to the query's relevant chunk ids.
    Hits and reciprocal ranks are aggregated into the returned report.
    """

    def __init__(
        self,
        loader: GoldenQueryLoader,
        retriever: Callable[[str], list[RetrievedChunk]],
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self._loader = loader
        self._retriever = retriever
        self._top_k = top_k

    def execute(self) -> RetrievalReport:
        queries = self._loader.load()
        if not queries:
            raise ValueError("golden query set must not be empty")
        evaluations = [self._evaluate(query) for query in queries]
        num_queries = len(evaluations)
        hit_rate = sum(evaluation.hit for evaluation in evaluations) / num_queries
        mean_reciprocal_rank = (
            sum(evaluation.reciprocal_rank for evaluation in evaluations)
            / num_queries
        )
        return RetrievalReport(
            num_queries=num_queries,
            hit_rate=hit_rate,
            mean_reciprocal_rank=mean_reciprocal_rank,
            per_query=evaluations,
        )

    def _evaluate(self, golden: GoldenQuery) -> QueryEvaluation:
        retrieved = self._retriever(golden.query)[: self._top_k]
        retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieved]
        relevant_chunk_ids = set(golden.relevant_chunk_ids)
        return QueryEvaluation(
            query=golden.query,
            relevant_chunk_ids=golden.relevant_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            top_k=self._top_k,
            hit=is_hit(retrieved_chunk_ids, relevant_chunk_ids),
            reciprocal_rank=reciprocal_rank(
                retrieved_chunk_ids, relevant_chunk_ids
            ),
        )