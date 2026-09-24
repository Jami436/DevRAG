# Architecture

DevRAG is an end-to-end Retrieval-Augmented Generation (RAG) platform. Source
documentation is parsed, chunked, embedded, and stored in PostgreSQL with the
pgvector extension. At query time, content is retrieved with hybrid search,
reranked, and used to generate grounded answers with source citations.

## Design principles

- **Hexagonal (ports-and-adapters) layout.** Business rules and interfaces live
  in `app/domain`; concrete technology lives in `app/infrastructure`; use-case
  workflows live in `app/application`; the HTTP surface lives in `app/api/v1`.
  Dependencies point inward: domain modules never import frameworks or
  infrastructure.
- **Provider swapping.** Embeddings, LLMs, rerankers, and health probes are
  selected at runtime from `app/core/settings.py`. Each provider implements a
  small interface defined in the domain, so implementations can be swapped or
  faked freely (see the offline/canned providers in `app/infrastructure`).
- **One configuration authority.** All environment-driven options are declared
  in `app/core/settings.py` (a `pydantic-settings` `BaseSettings`), read from
  `.env` and environment variables, and mutated nowhere else.

## Module map

```text
app/
├── api/v1/               FastAPI routes and Pydantic request/response schemas
│   ├── documents.py      POST/GET/DELETE /documents, GET /documents/{id}
│   ├── search.py         POST /search
│   ├── query.py          POST /query
│   └── health.py         GET /version, /health, /health/live, /health/ready
├── application/          Use-case workflows with injected dependencies
│   ├── ingestion/        IngestAndIndexDocument, IngestDocument
│   ├── retrieval/        RetrieveContext
│   ├── generation/       GenerateAnswer
│   └── evaluation/       EvaluateRetrieval, EvaluateGeneration
├── core/                 Settings, structured logging, middleware
├── domain/               Pure business logic
│   ├── documents/        Document/Chunk entities, repository interface
│   ├── embeddings/       EmbeddingProvider interface
│   ├── retrieval/        RetrievedChunk entity, RRF fusion, interfaces
│   ├── generation/       GeneratedAnswer, Citation, interfaces
│   ├── evaluation/       Metrics (hit rate, MRR, faithfulness, ...)
│   └── health/           Probe interface and report entity
├── infrastructure/       Concrete adapters
│   ├── db/               SQLAlchemy models, session, repositories
│   ├── ingestion/        Markdown/HTML/PDF parsers + registry, chunkers
│   ├── embeddings/       Local (sentence-transformers) and OpenAI providers
│   ├── llm/              OpenAI and canned providers
│   ├── reranking/        Cross-encoder, LLM, and identity rerankers
│   ├── health/           Database and API-key probes
│   └── evaluation/       JSON golden-query loader
└── services/             Higher-level services composed by the API layer
    ├── documents.py      DocumentService
    ├── search.py         SearchService
    └── query.py          QueryService
```

## Ingestion pipeline

1. **Parse.** A `ParserRegistry` dispatches on file extension to the Markdown,
   HTML, or PDF parser (`app/infrastructure/ingestion/`). Unknown types raise
   `UnknownFileTypeError`.
2. **Chunk.** Either the section chunker (splits on structural headings) or the
   token chunker (budget-aware, with overlap) produces `Chunk` objects.
3. **Embed.** An `EmbeddingProvider` maps each chunk to a vector.
4. **Store.** A `DocumentRepository` persists the document, its chunks, and
   their vectors in PostgreSQL/pgvector within a single transaction.

The use case `IngestAndIndexDocument` orchestrates these steps and is exposed
through `DocumentService` and the `POST /api/v1/documents` endpoint.

## Retrieval pipeline

1. **Hybrid search.** The repository executes a vector similarity search and a
   full-text (keyword) search over the same chunk set.
2. **Fusion.** Reciprocal Rank Fusion (RRF) combines the two ranked lists into a
   single ordering (`app/domain/retrieval/fusion.py`). Relative vector vs
   keyword weighting is configurable.
3. **Reranking.** A `Reranker` provider (cross-encoder, LLM, or identity)
   re-scores the fused candidates; the top `top_k` are returned.
4. **Search response.** `POST /api/v1/search` returns each hit with per-signal
   scores (`vector_score`, `keyword_score`), the fused/reranked
   `final_score`, and its 1-based `rank`.

## Generation pipeline

1. **Retrieve.** The query service retrieves and reranks the most relevant
   chunks (as above).
2. **Ground.** `GenerateAnswer` builds a prompt with instructions to cite
   sources inline with `[n]` markers, plus an excerpt budget per source.
3. **Map citations.** The model's inline `[n]` references are mapped back onto
   the retrieved chunks to produce structured `Citation` objects (chunk,
   document, excerpt, metadata).
4. **Query response.** `POST /api/v1/query` returns the answer text, the
   citations, and the model that generated it.

## Cross-cutting concerns

- **Health.** `/api/v1/health/live` answers whenever the process is up.
  `/api/v1/health/ready` runs every dependency probe (database, required API
  keys) and returns 200 or 503 with per-dependency results.
- **Logging.** `app/core/logging.py` emits structured logs (one JSON object per
  line, or plain text) for the root and uvicorn loggers.
- **Middleware.** `app/core/middleware.py` logs each completed request with
  method, path, status, duration, and client address.

## Settings

See `app/core/settings.py` and the [README configuration table](../README.md).

Key wiring variables: `EMBEDDING_PROVIDER`, `RERANKER_PROVIDER`,
`GENERATION_PROVIDER`, `EMBEDDING_DIMENSION` (must match the vector column),
`CHUNKING_STRATEGY`, and `DATABASE_URL`.