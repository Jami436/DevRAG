from app.domain.retrieval.entities import RetrievedChunk


class IdentityReranker:
    """Pluggable no-op reranker that keeps the incoming candidate order.

    Use it to disable reranking while keeping the retrieval pipeline wired
    through the same interface.
    """

    def rerank(
        self, query: str, results: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        del query
        ranked = list(results)
        for rank, hit in enumerate(ranked, start=1):
            hit.rank = rank
            if hit.final_score is None:
                hit.final_score = 0.0
        return ranked