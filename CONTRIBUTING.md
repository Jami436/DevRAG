# Contributing to DevRAG

Thanks for taking the time to contribute! DevRAG is a production-oriented RAG
platform for technical documentation. This guide explains how to set up a local
development environment, what conventions we follow, and how to get a change
merged.

## Table of Contents

- [Development environment](#development-environment)
- [Project layout](#project-layout)
- [Workflow](#workflow)
- [Code style](#code-style)
- [Testing](#testing)
- [Commits and pull requests](#commits-and-pull-requests)
- [Documentation](#documentation)
- [Code of Conduct](#code-of-conduct)

## Development environment

### Prerequisites

- Python 3.13+
- PostgreSQL 13+ with the `pgvector` extension

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Jami436/DevRAG.git
cd DevRAG

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install the package with all dev extras (test, lint, type)
pip install -e ".[dev]"

# 4. Create a local environment file from the defaults
#    (see app/core/settings.py for every option)
copy .env.example .env     # Windows
# cp .env.example .env     # macOS/Linux
```

If you only need the runtime dependencies, `pip install -e ".[test]"` keeps
lint/type tooling optional.

### Database

Create a PostgreSQL database and enable `pgvector`, then run the migrations:

```bash
alembic upgrade head
```

The default connection string is
`postgresql+psycopg://devrag:devrag@localhost:5432/devrag`; override it with
`DATABASE_URL` in `.env`.

## Project layout

```text
app/
├── api/v1/               FastAPI routes and request/response schemas
├── application/          Use-case workflows (ingest, retrieve, generate, evaluate)
├── core/                 Settings, logging, middleware
├── domain/               Pure business logic: entities, ports/interfaces, rules
├── infrastructure/       Concrete implementations: DB, parsers, embeddings, LLM, health
└── services/             Higher-level services composed by the HTTP layer
docs/                     Documentation pages and architecture decision records
scripts/                  Demo and evaluation runners
tests/                    Unit tests (plus integration/ for live-Postgres tests)
alembic/                  Database migrations
```

DevRAG follows a hexagonal (ports-and-adapters) architecture: `app/domain`
holds the business rules and interfaces, while `app/infrastructure` provides
concrete adapters. Keep domain modules free of framework imports.

## Workflow

1. **Fork** the repository and create a feature branch from `main`:
   `git checkout -b feat/your-change`.
2. Make focused commits with descriptive messages (see
   [Commits and pull requests](#commits-and-pull-requests)).
3. Run `ruff`, `mypy`, and the unit tests locally (below).
4. Open a pull request. The CI workflow runs lint, type-check, unit tests, and
   integration tests (against a `pgvector/pgvector:pg16` service container)
   automatically.

## Code style

- Python 3.13, `ruff` with the `E`, `F`, `I`, `B`, `SIM`, `UP`, `RUF` rule
  sets enabled and a line length of 88.
- `mypy` runs in strict mode with `pydantic.mypy` plugin support.
- Prefer type hints everywhere; annotate public functions and methods.
- Do not add comments unless they explain non-obvious intent.
- Match the existing conventions in the file you are changing.

Run the checks before pushing:

```bash
ruff check app tests scripts
mypy app
```

## Testing

Unit tests do not require external services:

```bash
pytest --ignore=tests/integration
```

Integration tests require a live PostgreSQL + pgvector instance:

```bash
pytest -m integration
```

When you add or change behavior, add or update tests in the same pull request.
Keep new tests small and focused on one behavior.

## Commits and pull requests

We use conventional-commit-style messages. The commit history mixes concise,
lowercase-style messages (for example `feat:`, `fix:`, `docs:`, `chore:`); try
to match that style:

- `feat: add batch ingestion endpoint`
- `fix: preserve citation order in generated answers`
- `docs: clarify evaluation dataset format`
- `chore: bump dev dependencies`

PR checklist:

- [ ] `ruff check app tests scripts` passes
- [ ] `mypy app` passes
- [ ] Unit tests pass (`pytest --ignore=tests/integration`)
- [ ] New behavior is covered by tests
- [ ] `CHANGELOG.md` updated under `Unreleased` (when user-facing)
- [ ] README / `docs/` updated if behavior or configuration changed

## Documentation

- User-facing configuration, API examples, and setup live in `README.md` and
  `docs/`.
- Significant architectural decisions are recorded as Architecture Decision
  Records (ADRs) in `docs/architecture-decision-records/`.

## Code of Conduct

Please note that this project has a
[Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to uphold
the standards described there.