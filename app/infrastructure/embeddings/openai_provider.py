from openai import OpenAI


class OpenAIEmbeddingProvider:
    """Generate embeddings through the OpenAI Embeddings API.

    Texts are sent in sub-batches capped at ``batch_size`` per request so large
    lists never exceed the API limit (2048 texts per call). The vector dimension
    is detected from the first embedding rather than assumed.
    """

    _DEFAULT_BATCH_SIZE = 2048

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        if not api_key:
            raise ValueError("An OpenAI API key is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._model = model
        self._batch_size = batch_size
        self._client = OpenAI(api_key=api_key)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            probe = self.embed(["probe"])
            self._dimension = len(probe[0])
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(embedding.embedding for embedding in response.data)
        return vectors