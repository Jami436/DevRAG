from sentence_transformers import CrossEncoder

from app.domain.retrieval.entities import RetrievedChunk


class CrossEncoderReranker:
    """Re-rank chunks on-device with a sentence-transformers cross-encoder.

    Every ``(query, chunk content)`` pair is scored by the transformer and the
    chunks are re-ordered by descending score. The model is loaded lazily so
    constructing the reranker never downloads weights or touches the GPU until
    the first call.
    """

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        if not model_name:
            raise ValueError("model_name cannot be empty")
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    def rerank(
        self, query: str, results: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not results:
            return []
        model = self._load()
        pairs = [(query, hit.content) for hit in results]
        raw_scores = model.predict(pairs).tolist()
        scored = list(zip(results, raw_scores, strict=True))
        scored.sort(key=lambda item: float(item[1]), reverse=True)
        for rank, (hit, score) in enumerate(scored, start=1):
            hit.final_score = float(score)
            hit.rank = rank
        return [hit for hit, _score in scored]

    def _load(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model