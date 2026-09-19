import json
from pathlib import Path
from uuid import UUID

from app.domain.evaluation.entities import GoldenQuery


class JSONGoldenQueriesLoader:
    """Load golden queries from a JSON document.

    The file must contain a list of entries shaped like::

        [
            {
                "query": "how does hybrid search work?",
                "relevant_chunk_ids": ["<uuid>", "<uuid>"]
            }
        ]
    """

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)

    def load(self) -> list[GoldenQuery]:
        if not self._file_path.exists():
            raise FileNotFoundError(
                f"Golden query file not found: {self._file_path}"
            )
        payload = json.loads(self._file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("golden query file must contain a JSON list")
        return [self._parse_entry(entry) for entry in payload]

    def _parse_entry(self, entry: object) -> GoldenQuery:
        if not isinstance(entry, dict):
            raise ValueError("each golden query must be a JSON object")
        query = entry.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError(
                "each golden query must have a non-empty 'query' string"
            )
        raw_ids = entry.get("relevant_chunk_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(
                "each golden query must have a non-empty 'relevant_chunk_ids' "
                "list"
            )
        relevant_chunk_ids = [
            self._parse_chunk_id(raw_id) for raw_id in raw_ids
        ]
        return GoldenQuery(
            query=query,
            relevant_chunk_ids=relevant_chunk_ids,
        )

    def _parse_chunk_id(self, raw_id: object) -> UUID:
        if not isinstance(raw_id, str):
            raise ValueError("chunk ids in 'relevant_chunk_ids' must be strings")
        try:
            return UUID(raw_id)
        except ValueError as error:
            raise ValueError(f"invalid chunk id {raw_id!r}") from error