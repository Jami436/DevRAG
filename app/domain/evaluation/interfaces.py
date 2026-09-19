from typing import Protocol

from app.domain.evaluation.entities import GoldenQuery


class GoldenQueryLoader(Protocol):
    """Load the golden query set that retrieval is evaluated against."""

    def load(self) -> list[GoldenQuery]:
        """Return the golden queries to evaluate."""