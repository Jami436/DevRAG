from sentence_transformers import SentenceTransformer


class LocalEmbeddingProvider:
    """Generate embeddings on-device with a sentence-transformers model.

    The model is loaded lazily so constructing the provider never downloads
    weights or touches the GPU until the first call. Encoding is batched to keep
    memory bounded on CPU and GPU alike. The vector dimension is read from the
    model itself after load.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        if not model_name:
            raise ValueError("model_name cannot be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            encoded = model.encode(batch)
            vectors.extend(row.tolist() for row in encoded)
        return vectors

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model