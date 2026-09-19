import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.infrastructure.evaluation.json_golden_queries_loader import (
    JSONGoldenQueriesLoader,
)


def _entry(resource: str, chunk_ids: list[object]) -> dict[str, object]:
    return {"query": resource, "relevant_chunk_ids": chunk_ids}


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_parses_valid_entries(tmp_path: Path) -> None:
    first = str(uuid4())
    second = str(uuid4())
    path = _write(
        tmp_path, [_entry("how does chunking work?", [first]), _entry("MRR?", [second])]
    )

    queries = JSONGoldenQueriesLoader(path).load()

    assert len(queries) == 2
    assert queries[0].query == "how does chunking work?"
    assert [str(chunk_id) for chunk_id in queries[0].relevant_chunk_ids] == [first]
    assert [str(chunk_id) for chunk_id in queries[1].relevant_chunk_ids] == [second]


def test_load_missing_file_raises(tmp_path: Path) -> None:
    loader = JSONGoldenQueriesLoader(tmp_path / "missing.json")

    with pytest.raises(FileNotFoundError, match="not found"):
        loader.load()


def test_load_rejects_non_list_document(tmp_path: Path) -> None:
    path = _write(tmp_path, {"query": "x"})

    with pytest.raises(ValueError, match="JSON list"):
        JSONGoldenQueriesLoader(path).load()


def test_load_rejects_non_object_entry(tmp_path: Path) -> None:
    path = _write(tmp_path, ["not an object"])

    with pytest.raises(ValueError, match="JSON object"):
        JSONGoldenQueriesLoader(path).load()


def test_load_rejects_missing_query(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"relevant_chunk_ids": [str(uuid4())]}])

    with pytest.raises(ValueError, match="'query'"):
        JSONGoldenQueriesLoader(path).load()


def test_load_rejects_empty_query(tmp_path: Path) -> None:
    path = _write(tmp_path, [_entry("", [str(uuid4())])])

    with pytest.raises(ValueError, match="'query'"):
        JSONGoldenQueriesLoader(path).load()


def test_load_rejects_missing_chunk_ids(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"query": "query"}])

    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        JSONGoldenQueriesLoader(path).load()


def test_load_rejects_empty_chunk_ids(tmp_path: Path) -> None:
    path = _write(tmp_path, [_entry("query", [])])

    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        JSONGoldenQueriesLoader(path).load()


def test_load_rejects_non_string_chunk_id(tmp_path: Path) -> None:
    path = _write(tmp_path, [_entry("query", [123])])

    with pytest.raises(ValueError, match="must be strings"):
        JSONGoldenQueriesLoader(path).load()


def test_load_rejects_invalid_uuid(tmp_path: Path) -> None:
    path = _write(tmp_path, [_entry("query", ["not-a-uuid"])])

    with pytest.raises(ValueError, match="invalid chunk id"):
        JSONGoldenQueriesLoader(path).load()
