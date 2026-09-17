from typing import Protocol


class EmbeddingProvider(Protocol):
    """Generate dense vector embeddings for text inputs."""

    @property
    def dimension(self) -> int:
        """Return the number of dimensions in every generated vector."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return one vector per text, aligned by index."""