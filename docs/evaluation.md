# Evaluation

DevRAG ships an evaluation harness that measures both retrieval quality and
generation quality against a golden query set. It writes a JSON report to
`data/evaluation/report.json`.

## Metrics

### Retrieval

- **Hit rate** — the fraction of queries for which at least one relevant chunk
  appears in the top `top_k` results.
- **Mean Reciprocal Rank (MRR)** — the average of `1 / rank` of the first
  relevant chunk, averaged over queries.

### Generation

- **Faithfulness** — whether the generated answer is supported by the retrieved
  context.
- **Answer relevancy** — whether the answer actually addresses the question.
- **Context precision / relevancy** — how much of the retrieved context is
  relevant to the question and used in the answer.

## Golden query set

The golden set lives in `data/evaluation/golden_queries.json`. Each entry maps a
natural-language question to the chunk UUIDs considered ground truth:

```json
[
  {
    "query": "How does hybrid search work?",
    "relevant_chunk_ids": ["12345678-1234-1234-1234-123456789012"]
  }
]
```

`relevant_chunk_ids` must be a non-empty list of UUIDs matching real indexed
chunks in the database (see `data/evaluation/README.md` for how to populate it
after ingestion).

## Running evaluation

```bash
# Retrieval + generation metrics (requires an LLM provider configured)
python scripts/evaluate.py

# Retrieval metrics only
python scripts/evaluate.py --skip-generation

# Custom inputs / outputs
python scripts/evaluate.py --golden-json PATH --report PATH --top-k N
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--golden-json PATH` | `evaluation_golden_queries_path` | Golden queries file |
| `--report PATH` | `data/evaluation/report.json` | Output report path |
| `--top-k N` | `retriever_top_k` (5) | Chunks considered per query |
| `--skip-generation` | off | Skip faithfulness / answer relevancy |

Evaluation requires a live PostgreSQL with pgvector and works with either the
OpenAI or the local embedding provider. Generation metrics additionally require
an LLM provider (`GENERATION_PROVIDER=openai` with an API key, or `none` for
the offline canned provider).

## Report

The report aggregates both retrieval and generation metrics across all golden
queries. It is written as JSON under `data/evaluation/` so it can be stored,
diffed across runs, and fed into dashboards.

See also: `data/evaluation/README.md` for the dataset format and population
steps.