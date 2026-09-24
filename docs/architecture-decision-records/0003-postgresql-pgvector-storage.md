# ADR-0003: PostgreSQL + pgvector for indexed storage

- **Status:** Accepted
- **Date:** 2026-09-24

## Context

DevRAG needs to store documents and their chunks alongside high-dimensional
embedding vectors, and to retrieve chunks by both vector similarity and keyword
matching to support hybrid search. The options considered were a dedicated
vector database (e.g. Milvus, Qdrant, Pinecone), Elasticsearch/OpenSearch, or
PostgreSQL with the pgvector extension.

## Decision

Store documents, chunks, and vectors in **PostgreSQL with the pgvector
extension**, using SQLAlchemy models and Alembic migrations.

Rationale:

- A single operational store: relational metadata, chunks, and vectors share
  transactions, backups, and access control; ingestion updates a document and
  its chunks atomically.
- Native SQL **keyword (full-text) search** coexists with vector search, which
  directly supports hybrid retrieval (see ADR-0004) without a second system.
- Mature infrastructure and tooling; the team already operates Postgres.
- pgvector keeps the deployment surface small for a docs-RAG product.

Vector dimensions are set at migration time and must match `EMBEDDING_DIMENSION`
in settings (1536 for OpenAI, 384 for the local sentence-transformers model).

## Consequences

- **Positive:** single source of truth for all retrieval state; transactional
  consistency; simpler ops; cheaper than a second data store.
- **Negative:** Postgres + pgvector does not scale to very large corpora the way
  dedicated vector stores do; ANN recall is approximate.
- **Risk:** schema changes must be coordinated with embedding dimensions —
  mitigated by Alembic migrations and an explicit configuration contract.