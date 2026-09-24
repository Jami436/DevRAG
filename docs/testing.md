# Testing

DevRAG uses `pytest`. Tests are split into unit tests (no external services)
and integration tests (require a live PostgreSQL with pgvector).

## Running tests

Install the dev extras first:

```bash
pip install -e ".[dev]"
```

### Unit tests

```bash
pytest --ignore=tests/integration
```

These cover parsing, chunking, embedding providers, retrieval fusion, rerankers,
generation and citation mapping, repositories (against fakes), services, API
routes (via `TestClient`), settings, logging, and health probes.

### Integration tests

Integration tests require a PostgreSQL instance with the `pgvector` extension
and the migrations applied (`alembic upgrade head`). They are tagged with the
`integration` marker and must be selected explicitly:

```bash
pytest -m integration
```

They connect using the `DATABASE_URL` from settings and cover the document
repository, retrieval evaluation, and search against real Postgres.

Run the whole suite (unit + integration) with:

```bash
pytest
```

## Test layout

```text
tests/
├── conftest.py                    Shared fixtures (client, session, fakes)
├── test_*.py                      One module per behavior area
└── integration/
    ├── test_document_repository_postgres.py
    ├── test_retrieval_evaluation_postgres.py
    └── test_search_postgres.py
```

The `integration` marker is declared in `pyproject.toml` with its meaning so
`pytest --markers` documents it.

## Lint and type-check

```bash
ruff check app tests scripts
mypy app
```

`ruff` enforces the `E`, `F`, `I`, `B`, `SIM`, `UP`, `RUF` rule sets with a
line length of 88. `mypy` runs in strict mode with the pydantic plugin.

## CI

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs four jobs on push
to `main` and pull requests:

1. **lint** — `ruff check app tests scripts`
2. **typecheck** — `mypy app`
3. **unit-tests** — `pytest --ignore=tests/integration`
4. **integration-tests** — spins up a `pgvector/pgvector:pg16` service
   container, sets `DATABASE_URL`, and runs `pytest -m integration`

A pull request must pass all four jobs before it is merged.