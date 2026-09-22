# DevRAG

A production-oriented Retrieval-Augmented Generation (RAG) platform designed specifically for technical and developer documentation. DevRAG provides end-to-end capabilities for ingesting, indexing, retrieving, and generating grounded answers from technical documentation using modern LLM and vector search technologies.

## Overview

DevRAG is under active development with a working end-to-end RAG pipeline: documentation is **parsed**, **chunked**, **embedded**, and **stored** in PostgreSQL + pgvector, then retrieved with **hybrid search**, **reranked**, and used to generate **grounded answers with source citations**. A REST API exposes ingestion, search, query, and health endpoints. The architecture keeps domain logic independent of infrastructure via injected ports, so providers can be swapped or faked freely.

## Features

- **Ingestion pipeline** — parse -> chunk -> embed -> store in one end-to-end flow
  - Parsers for **Markdown**, **HTML**, and **PDF** (PyMuPDF), selected by file extension
  - Chunkers: **section-aware** and **token-based** (with configurable overlap)
  - Embedding providers: **OpenAI API** or **local on-device** (sentence-transformers)
- **Storage** — PostgreSQL with **pgvector** for vector similarity, Alembic migrations, and a repository pattern with idempotent re-ingestion (upserts keyed by `document_id` + `chunk_index`)
- **Hybrid retrieval** — dense vector + PostgreSQL full-text search fused with **Reciprocal Rank Fusion (RRF)**
- **Reranking** — pluggable rerankers: **cross-encoder** (on-device), **LLM-based** (OpenAI chat scoring), or **identity** (no-op)
- **Grounded answer generation** — the LLM answers strictly from retrieved passages and `[n]` references are mapped back to structured **citations** (chunk id, page, excerpt)
- **Retrieval evaluation** — **hit rate** and **mean reciprocal rank (MRR)** against a JSON golden-query set
- **REST API** — document upload/list/get/delete, search, question answering, and liveness/readiness/version probes
- **Structured logging** — stdlib `logging` emitting JSON lines to stdout (configurable log level and text/JSON format), plus a middleware that logs one structured record per HTTP request
- **256 unit tests** covering the full stack, plus optional integration tests against live PostgreSQL + pgvector (264 total)
- **Strict quality gates** — ruff linting, mypy strict typing, pytest, Python 3.13

## Architecture

DevRAG follows a layered, hexagonal-style architecture:

- **`app/domain/`** — Core business concepts and ports (interfaces) independent of frameworks: documents, chunks, retrieval, reranking, generation/citations, evaluation, embeddings, repositories, health
- **`app/application/`** — Use-case workflows that orchestrate domain ports: `IngestAndIndexDocument`, `RetrieveContext`, `GenerateAnswer`, `EvaluateRetrieval`
- **`app/infrastructure/`** — Concrete technology implementations: parsers, chunkers, embedding adapters, SQL/pgvector repository, rerankers, LLM providers, health probes, DB session
- **`app/services/`** — Higher-level services that wire the default configurations and expose reusable endpoints (`SearchService`, `QueryService`, `DocumentService`)
- **`app/api/v1/`** — FastAPI routes and Pydantic request/response schemas
- **`app/core/`** — Central `Settings` (env-driven via pydantic-settings), app config, and structured logging setup

## Project Structure

```text
devrag/
├── app/
│   ├── api/v1/                 # FastAPI routers: documents, search, query, health
│   ├── core/                   # Settings / configuration / logging & middleware
│   ├── domain/                 # Entities, ports/interfaces, pure logic (fusion, metrics, citations)
│   ├── application/            # Use cases: ingest, retrieve, generate, evaluate
│   ├── infrastructure/         # Parsers, chunkers, embeddings, rerankers, LLM, DB, health probes
│   └── services/               # Wiring and reusable service layer
├── tests/
│   └── integration/            # Tests requiring live PostgreSQL + pgvector
├── alembic/                    # Migrations: 001 initial schema, 002 FTS index
├── data/
│   ├── raw/                    # Source documentation
│   ├── processed/              # Chunked/intermediate artifacts
│   └── evaluation/             # golden_queries.json for retrieval evaluation
├── .github/workflows/          # CI/CD (stub)
├── pyproject.toml
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.13+
- PostgreSQL 13+ with the `pgvector` extension (for the vector store and integration tests)

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate       # on Windows
# source .venv/bin/activate  # on macOS/Linux

# Install the package with dev dependencies
pip install -e ".[dev]"
```

### Configuration

Copy `.env` (or set the following environment variables) — see `app/core/settings.py` for all options:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+psycopg://devrag:devrag@localhost:5432/devrag` | SQLAlchemy connection string |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `local` |
| `OPENAI_API_KEY` | — | Required when using OpenAI embeddings / LLM generation / LLM reranker |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model id |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | On-device model (falls back to 384-dim) |
| `EMBEDDING_DIMENSION` | `1536` | Vector dimension; must match the migration's `Vector(1536)` |
| `CHUNKING_STRATEGY` | `section` | `section` or `token` |
| `CHUNKING_TOKEN_LIMIT` | `512` | Max tokens per chunk |
| `RETRIEVER_TOP_K` | `5` | Chunks returned after reranking |
| `RERANKER_PROVIDER` | `cross_encoder` | `cross_encoder`, `llm`, or `none` |
| `GENERATION_PROVIDER` / `GENERATION_MODEL` | `openai` / `gpt-4o-mini` | LLM used to generate answers |
| `EVALUATION_GOLDEN_QUERIES_PATH` | `data/evaluation/golden_queries.json` | Golden queries for eval |
| `LOG_LEVEL` | `INFO` | Minimum severity for the root and uvicorn loggers |
| `LOG_FORMAT` | `json` | Output format: `json` (structured, default) or `text` |

### Database setup

```bash
# Create the database and enable pgvector, then apply migrations
alembic upgrade head
```

### Running the API

```bash
# Launch with structured JSON logging enabled (uses app/core/logging.py)
python -m app.main

# Or launch directly with uvicorn (default uvicorn log format applies unless --log-config is supplied)
uvicorn app.main:app --reload
```

Open the interactive docs at <http://127.0.0.1:8000/docs>.

Logs are emitted to stdout as one JSON object per line, e.g.:

```json
{"ts": "2026-09-21T10:00:00.000000Z", "level": "INFO", "logger": "app.access", "message": "request completed", "module": "middleware", "function": "request_logging_middleware", "line": 35, "method": "POST", "path": "/api/v1/search", "status": 200, "duration_ms": 42.5, "client": "127.0.0.1"}
```

Attach extra structured fields to any log call with the stdlib `extra` kwarg:
`logger.info("chunk embedded", extra={"document_id": id, "chunks": n})`. To keep
uvicorn's own logging from replacing this configuration, run the service with
`python -m app.main`; the `uvicorn`/`uvicorn.error`/`uvicorn.access` loggers are
configured from `LOG_LEVEL` / `LOG_FORMAT` in the same setup.

## API

All routes are mounted under `/api/v1`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/documents` | Upload a Markdown/HTML/PDF file to ingest and index it |
| `GET` | `/documents` | List documents (paginated) |
| `GET` | `/documents/{document_id}` | Document metadata plus its indexed chunks |
| `DELETE` | `/documents/{document_id}` | Delete a document and its chunks |
| `POST` | `/search` | Hybrid retrieval + rerank; returns per-signal scores and ranks |
| `POST` | `/query` | End-to-end question answering with source citations |
| `GET` | `/version` | Running service name and version |
| `GET` | `/health` | Liveness summary (compat) |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe verifying all dependencies (200/503) |

### Example: answer a question

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What algorithm does the iris project use?", "top_k": 5}'
```

## Tests

```bash
# Unit tests (no external services required)
pytest --ignore=tests/integration

# Integration tests (requires DATABASE_URL pointing at a live PostgreSQL + pgvector)
pytest -m integration
```

Lint and type-check with the dev extras:

```bash
ruff check .
mypy app
```

## Development Status

| Area | Status |
|------|--------|
| Ingestion (parse / chunk / embed / store) | Implemented |
| Parser registry (Markdown, HTML, PDF) | Implemented |
| Chunking (section, token) | Implemented |
| Storage (SQLAlchemy + pgvector, Alembic) | Implemented |
| Hybrid retrieval + RRF fusion | Implemented |
| Reranking (cross-encoder, LLM, identity) | Implemented |
| Grounded generation with citations | Implemented |
| Retrieval evaluation (hit rate, MRR) | Implemented |
| REST API + health/readiness probes | Implemented |
| Structured logging (JSON / text, access logs) | Implemented |
| CI/CD workflows, Docker deployment | Planned |
| Background processing, caching, observability | Planned |

## Roadmap

- Docker-compose for local Postgres + pgvector, and containerized deployment
- GitHub Actions CI (lint, type-check, unit + integration tests)
- Background ingestion jobs, caching, and tracing/metrics (structured logging is done)
- Additional parsers (reStructuredText, docx) and chunkers
- Answer generation evaluation (faithfulness/groundedness metrics)

## License

[MIT](LICENSE)