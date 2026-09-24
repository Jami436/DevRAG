# DevRAG Documentation

Welcome to the DevRAG documentation. DevRAG is a production-oriented
Retrieval-Augmented Generation (RAG) platform for technical and developer
documentation.

The quickest way to get started is the [README](../README.md). These pages go
deeper into how the system is built, how to operate it, and the decisions
behind it.

## Guides

- [Architecture](architecture.md) — layers, modules, and data flow
- [API Reference](api.md) — REST endpoints, schemas, and examples
- [Testing](testing.md) — how unit and integration tests are organized and run
- [Evaluation](evaluation.md) — retrieval and generation quality harness
- [Deployment](deployment.md) — running DevRAG in production

## Architecture Decision Records

Significant technical decisions are recorded as ADRs. Each entry states the
context, the decision, and the consequences.

1. [ADR-0001: Record architecture decisions](architecture-decision-records/0001-record-architecture-decisions.md)
2. [ADR-0002: Hexagonal (ports-and-adapters) layout](architecture-decision-records/0002-hexagonal-architecture.md)
3. [ADR-0003: PostgreSQL + pgvector for indexed storage](architecture-decision-records/0003-postgresql-pgvector-storage.md)
4. [ADR-0004: Hybrid retrieval with reciprocal rank fusion and reranking](architecture-decision-records/0004-hybrid-retrieval-reciprocal-rank-fusion.md)
5. [ADR-0005: Grounded generation, observability, and evaluation](architecture-decision-records/0005-grounded-generation-observability-evaluation.md)

## Project conventions

- [Contributing](../CONTRIBUTING.md) — setting up a dev environment and merging changes
- [Code of Conduct](../CODE_OF_CONDUCT.md)
- [Changelog](../CHANGELOG.md)
- [License](../LICENSE)