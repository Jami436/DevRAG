# API Reference

DevRAG exposes a REST API under `/api/v1`. The default base URL is
`http://127.0.0.1:8000`. Interactive documentation is available at
`/docs` (Swagger UI) and `/redoc`.

All request and response bodies are JSON. List and detail responses use UUIDs
for document and chunk identifiers.

## Endpoints

| Method | Path    | Description |
|--------|---------|-------------|
| POST   | `/api/v1/documents` | Ingest an uploaded documentation file (multipart) |
| GET    | `/api/v1/documents` | List indexed documents (paginated, newest first) |
| GET    | `/api/v1/documents/{document_id}` | Document metadata and its indexed chunks |
| DELETE | `/api/v1/documents/{document_id}` | Delete a document and its chunks |
| POST   | `/api/v1/search` | Retrieve and rerank relevant chunks |
| POST   | `/api/v1/query` | Generate a grounded answer with source citations |
| GET    | `/api/v1/version` | Service name and version |
| GET    | `/api/v1/health` | Backward-compatible liveness summary |
| GET    | `/api/v1/health/live` | Liveness probe |
| GET    | `/api/v1/health/ready` | Readiness probe (all dependencies) |

## Ingest a document

`POST /api/v1/documents` — multipart upload. Markdown, HTML, and PDF files are
supported. Returns **201** with the created document, **415** for unsupported
file types.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -H "Content-Type: multipart/form-data" \
  -F "file=@docs/guide.md"
```

Response:

```json
{
  "id": "f1a6b0c2-...-c9d8e7f6a5b4",
  "title": "guide.md",
  "source": "upload",
  "source_url": null,
  "metadata": {},
  "created_at": "2026-09-24T10:00:00Z"
}
```

## List documents

`GET /api/v1/documents?limit=20&offset=0` — returns `items`, `total`, `limit`,
`offset`.

```bash
curl -G http://127.0.0.1:8000/api/v1/documents \
  --data-urlencode "limit=10"
```

## Get a document

`GET /api/v1/documents/{document_id}` — returns the document plus a `chunks`
array. Returns **404** if the document does not exist.

```bash
curl http://127.0.0.1:8000/api/v1/documents/f1a6b0c2-...-c9d8e7f6a5b4
```

## Delete a document

`DELETE /api/v1/documents/{document_id}` — returns **204** on success, **404**
if the document is not found.

```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/documents/f1a6b0c2-...-c9d8e7f6a5b4
```

## Search

`POST /api/v1/search` — hybrid retrieval with reranking.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What algorithm does the iris project use?", "top_k": 5}'
```

Request fields:

| Field   | Type    | Default | Constraints |
|---------|---------|---------|-------------|
| `query` | string  | —       | 1–4096 chars |
| `top_k` | integer | `retriever_top_k` (5) | 1–20 |

Response: a `results` array; each hit carries `chunk_id`, `document_id`,
`document_title`, `content`, `chunk_index`, `page_number`, `metadata`, the
per-signal scores `vector_score` and `keyword_score`, the fused/reranked
`final_score`, and the 1-based `rank`.

```json
{
  "query": "What algorithm does the iris project use?",
  "results": [
    {
      "chunk_id": "a1b2c3d4-...",
      "document_id": "f1a6b0c2-...",
      "document_title": "chunking.md",
      "content": "...",
      "chunk_index": 3,
      "page_number": 1,
      "metadata": {},
      "vector_score": 0.31,
      "keyword_score": 0.0,
      "final_score": 0.045,
      "rank": 1
    }
  ]
}
```

## Query (grounded answers)

`POST /api/v1/query` — retrieve, rerank, and generate a grounded answer with
source citations.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What algorithm does the iris project use?", "top_k": 5}'
```

Request fields are the same as `/search`. Response: `query`, `answer`, an
`citations` array, and `model`.

Each citation contains `chunk_id`, `document_id`, `chunk_index`,
`page_number`, `document_title`, `excerpt`, and `metadata`.

## Health and version

```bash
curl http://127.0.0.1:8000/api/v1/version
# {"service": "DevRAG", "version": "0.1.0"}

curl http://127.0.0.1:8000/api/v1/health/live
# {"status": "ok", "service": "DevRAG", "version": "0.1.0"}

curl http://127.0.0.1:8000/api/v1/health/ready
# 200: {"status": "ready", "service": "DevRAG", "version": "0.1.0", "dependencies": [...]}
# 503: {"status": "unhealthy", ..., "dependencies": [{"service": "database", "state": "error", ...}]}
```

`/health/ready` returns **200** only when every dependency probe (database,
required API keys) reports `ok`, and **503** otherwise, with per-dependency
details useful to operators.

## Status codes

| Code | Meaning |
|------|---------|
| 201  | Document ingested |
| 204  | Document deleted |
| 404  | Document not found |
| 415  | Unsupported file type |
| 422  | Validation error (bad request body) |
| 503  | Readiness probe: a dependency is unavailable |