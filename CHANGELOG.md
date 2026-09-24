# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Documentation: this changelog, contributing guide, code of conduct, docs
  pages, and architecture decision records.
- Deployment assets: container images and deployment guide.

## [0.1.0] - 2026-09-24

### Added

- **Foundations**
  - Application skeleton, settings via `pydantic-settings` (`.env`), and a
    health endpoint.
  - Domain model with page-level document/chunk structure.
  - Linting (ruff), strict typing (mypy + pydantic plugin), and test tooling.
- **Ingestion**
  - Markdown, HTML, and PDF parsers behind an extension-keyed parser registry.
  - Section and token chunkers with configurable limits and overlap.
  - Embedding providers (OpenAI and local sentence-transformers) behind a common
    interface and factory.
  - SQLAlchemy + pgvector storage layer, repository pattern, and Alembic
    migrations (initial schema, full-text index).
  - End-to-end ingest-and-index pipeline (parse -> chunk -> embed -> store).
- **Retrieval & reranking**
  - Hybrid retrieval combining vector and keyword search.
  - Reciprocal Rank Fusion (RRF) for combining ranked lists.
  - Reranking layer: cross-encoder, LLM-based, and identity providers.
- **Evaluation**
  - Retrieval evaluation harness with hit rate and MRR.
  - Golden-query dataset (`data/evaluation/golden_queries.json`) and JSON loader.
- **Generation**
  - Grounded answer generation with inline `[n]` citations mapped to source
    chunks.
  - OpenAI and offline canned LLM providers.
- **API**
  - Complete REST API: document upload/list/get/delete, search, and query.
  - Liveness/readiness probes, version endpoint, and dependency checks.
- **Observability**
  - Structured logging (JSON/text) and request-access middleware.
  - Generation evaluation metrics (faithfulness, answer relevancy, context
    precision/relevancy) and `scripts/evaluate.py` report runner.
- **CI/CD**
  - GitHub Actions workflow: lint, type-check, unit tests, and integration
    tests against a `pgvector/pgvector:pg16` service container.
- **Docs & demo**
  - README with architecture overview, quickstart, API examples, and settings
    table.
  - `scripts/demo.py` end-to-end demo with bundled sample docs in `data/raw/`.

[Unreleased]: https://github.com/Jami436/DevRAG/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Jami436/DevRAG/releases/tag/v0.1.0