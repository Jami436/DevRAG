# Golden Evaluation Dataset

This directory contains the golden query set used for retrieval and generation evaluation. Each entry is a JSON object with:

- ``query``: a natural-language question about the documentation
- ``relevant_chunk_ids``: the chunk IDs (UUIDs) considered **ground truth** for that query

The file is loaded by :class:`app.infrastructure.evaluation.JSONGoldenQueriesLoader`.

## Format

````json
[
  {
    "query": "How does hybrid search work?",
    "relevant_chunk_ids": ["<uuid>", "<uuid>"]
  }
]
````

The ``relevant_chunk_ids`` must be a **non-empty** list of valid UUID strings. These IDs must match the ``chunk_id`` values of the actual indexed chunks in the database.

## Populating the Dataset

1. **Ingest documentation** using the DevRAG REST API or CLI. This creates chunks with real UUIDs in the PostgreSQL + pgvector database.

2. **Query the document repository** to retrieve the actual chunk IDs for your topics, e.g.:

   ```python
   from app.infrastructure.db.repositories.document_repository import DocumentRepository
   from app.infrastructure.db.session import SessionLocal
   from uuid import UUID

   session = SessionLocal()
   repo = DocumentRepository(session)
   # ... find chunks related to your topic ...
   chunk_ids = [str(c.chunk_id) for c in chunks]
   session.close()
   ```

3. **Update ``data/evaluation/golden_queries.json``** with the real chunk IDs obtained in step 2. The JSON format must remain a JSON list of entries, each with a ``query`` string and a ``relevant_chunk_ids`` list of UUID string values.

4. **Run the evaluation script**:

   ```bash
   python scripts/evaluate.py
   ```

   This will compute retrieval metrics (hit rate, MRR) and generation metrics
   (faithfulness, answer relevancy, context precision/relevancy) and write a
   report to ``data/evaluation/report.json``.

## Notes

- The placeholder UUIDs in the initial ``golden_queries.json`` are **not**
  valid for any real dataset and must be replaced after ingestion.
- If you wish to run evaluation without generating answers (retrieval-only),
  use the ``--skip-generation`` flag on the eval runner.
- The evaluation runner can be run against a live PostgreSQL + pgvector
  instance by setting ``DATABASE_URL`` and, optionally, ``OPENAI_API_KEY``
  for generation.