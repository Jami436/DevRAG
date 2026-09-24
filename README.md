# DevRAG

A production-oriented Retrieval-Augmented Generation (RAG) platform for technical and developer documentation. DevRAG provides end-to-end capabilities for ingesting, indexing, retrieving, and generating grounded answers from technical documentation using modern LLM and vector search technologies.

![DevRAG Architecture Diagram](https://raw.githubusercontent.com/Jami436/DevRAG/main/docs/architecture-diagram.svg)

## Overview

DevRAG is an active RAG pipeline: documentation is **parsed**, **chunked**, **embedded**, and **stored** in PostgreSQL + pgvector, then retrieved with **hybrid search**, **reranked**, and used to generate **grounded answers with source citations**. A REST API exposes ingestion, search, query, and health endpoints. The architecture keeps domain logic independent of infrastructure via injected ports, so providers can be swapped or faked freely.

## Architecture

DevRAG follows a layered, hexagonal (ports-and-adapters) architecture:

```text
+-----------------------------+      +---------------------------+
|          app/api/v1/         |      |    app/core/              |
|  FastAPI routes + schemas   |<-->  |  Settings / config /      |
+-----------------------------+      |  logging / middleware     |
          ^                           +---------------------------+
          |                                   ^
          |                                   |
+-----------------------------+      +---------------------------+
|     app/services/           |      | app/domain/               |
|  Higher-level services      |      |  Core business concepts   |
|  (SearchService,           |      |  Entities, ports/interfaces|
|   QueryService,           |      |  (pure logic, no frameworks|
|   DocumentService)        |      |   — no framework imports) |
+-----------------------------+      +---------------------------+
          |                                   |
          |                                   |
+-----------------------------+      +---------------------------+
|   app/application/          |      | app/infrastructure/       |
|  Use-case workflows:       |      |  Concrete tech implementations|
|  IngestAndIndexDocument,   |      |  Parsers, chunkers,       |
|  RetrieveContext,         |      |   embeddings, rerankers,  |
|  GenerateAnswer,          |      |   SQL/pgvector repos,     |
|  EvaluateRetrieval        |      |   health probes, LLM      |
+-----------------------------+      +---------------------------+
```

## Quickstart

### Prerequisites

- Python 3.13+
- PostgreSQL 13+ with the `pgvector` extension

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate  # on Windows
# source .venv/bin/activate  # on macOS/Linux

# Install the package with dev dependencies
pip install -e ".[dev]"
```

### Configuration

Copy `.env` (or set environment variables) — see `app/core/settings.py` for all options:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://devrag:devrag@localhost:5432/devrag` | SQLAlchemy connection string |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `local` |
| `OPENAI_API_KEY` | — | Required for OpenAI embeddings/LLM |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model id |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | On-device model (384-dim) |
| `EMBEDDING_DIMENSION` | `1536` | Vector dimension; must match migration |
| `CHUNKING_STRATEGY` | `section` | `section` or `token` |
| `CHUNKING_TOKEN_LIMIT` | `512` | Max tokens per chunk |
| `RETRIEVER_TOP_K` | `5` | Chunks returned after reranking |
| `RERANKER_PROVIDER` | `cross_encoder` | `cross_encoder`, `llm`, or `none` |
| `GENERATION_PROVIDER` / `GENERATION_MODEL` | `openai` / `gpt-4o-mini` | LLM for answer generation |
| `LOG_LEVEL` | `INFO` | Minimum severity for root and uvicorn loggers |
| `LOG_FORMAT` | `json` | Output format: `json` or `text` |

### Database setup

```bash
# Create the database and enable pgvector, then apply migrations
alembic upgrade head
```

### Running the API

```bash
# Launch with structured JSON logging (recommended)
python -m app.main

# Or launch directly with uvicorn
uvicorn app.main:app --reload
```

Open the interactive docs at <http://127.0.0.1:8000/docs>.

Logs are emitted to stdout as one JSON object per line:

```json
{"ts": "2026-09-21T10:00:00.000000Z", "level": "INFO", "logger": "app.access", "message": "request completed", "module": "middleware", "function": "request_logging_middleware", "line": 35, "method": "POST", "path": "/api/v1/search", "status": 200, "duration_ms": 42.5, "client": "127.0.0.1"}
```

## API Examples

All routes are mounted under `/api/v1`.

### Upload and ingest a documentation file

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -F "file=@/path/to/docs.md"
```

### Search for relevant chunks

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What algorithm does the iris project use?", "top_k": 5}'
```

### Answer a question with source citations

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What algorithm does the iris project use?", "top_k": 5}'
```

### List documents

```bash
curl -G http://127.0.0.1:8000/api/v1/documents \
  --data-urlencode "limit=10"
```

### Delete a document

```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/documents/{document_id}
```

### Health probes

```bash
curl http://127.0.0.1:8000/api/v1/health/live
curl http://127.0.0.1:8000/api/v1/health/ready
curl http://127.0.0.1:8000/api/v1/version
```

## Demo

`scripts/demo.py` ingests a real documentation set and shows a grounded answer
end to end:

```bash
python scripts/demo.py                       # ingest data/raw and answer the default questions
python scripts/demo.py --query "What is reciprocal rank fusion?" --top-k 5
python scripts/demo.py --no-generation       # retrieval only
```

It walks a docs directory (default `data/raw`), ingests every Markdown/HTML/PDF
file through the same parse → chunk → embed → store pipeline the API uses, then
prints the hybrid-retrieval hits with per-signal scores and the grounded answer
with source citations. It honours `EMBEDDING_PROVIDER`, `RERANKER_PROVIDER` and
`GENERATION_PROVIDER` from settings, so it works with OpenAI or offline
providers. Generation requires `GENERATION_PROVIDER=openai` plus
`OPENAI_API_KEY` (or `none` for the offline canned provider).

Sample docs covering hybrid search, chunking, reciprocal rank fusion and
grounded answers ship under `data/raw/` and double as demo assets.

## Demo Assets

### Data directory structure

```
data/
├── raw/                    # Source documentation (Markdown, HTML, PDF) + demo set
│   ├── chunking.md
│   ├── grounded-answers.md
│   ├── hybrid-retrieval.md
│   └── reciprocal-rank-fusion.md
├── processed/              # Chunked/intermediate artifacts
│   └── .gitkeep
└── evaluation/
    ├── golden_queries.json  # Golden query set for retrieval/eval
    └── README.md            # Evaluation dataset format
```

### Golden evaluation dataset format

Each entry in `data/evaluation/golden_queries.json` is a JSON object:

```json
[
  {
    "query": "What is hybrid search and how does it work?",
    "relevant_chunk_ids": ["12345678-1234-1234-1234-123456789012"]
  }
]
```

The `relevant_chunk_ids` must be valid UUID strings matching indexed chunks in the database. Placeholder UUIDs in the initial file must be replaced after ingestion.

### Evaluation

```bash
# Run retrieval evaluation (hit rate, MRR)
python -m pytest tests/ -m integration --tb=short

# Or run via script
python scripts/evaluate.py
```

This computes retrieval metrics and generation metrics (faithfulness, answer relevancy, context precision/relevancy) and writes a report to `data/evaluation/report.json`.

## Tests

```bash
# Unit tests (no external services required)
pytest --ignore=tests/integration

# Integration tests (requires live PostgreSQL + pgvector)
pytest -m integration
```

Lint and type-check with dev extras:

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
| GitHub Actions CI (lint, type-check, unit + integration) | Implemented |
| Docker deployment (image, Compose, deployment guide) | Implemented |
| Background processing, caching, observability | Planned |

## Roadmap

- Background ingestion jobs, caching, and tracing/metrics
- Additional parsers (reStructuredText, docx) and chunkers
- Packaging tuning and additional deployment targets (Kubernetes helm chart)
- Generation evaluation refinements and golden-dataset expansion

## License

[MIT](LICENSE)