# ADR-0005: Grounded generation, observability, and evaluation

- **Status:** Accepted
- **Date:** 2026-09-24

## Context

DevRAG exists to give *answerable with sources* answers from documentation.
Blind LLM generation would risk hallucination with no way to verify claims.
Operating the service in production additionally requires signals about health,
structured diagnostics, and a way to measure whether retrieval and generation
are actually improving.

## Decision

Adopt four cross-cutting decisions:

### 1. Grounded generation with source citations

The generator (`app/application/generation/generate_answer.py`) receives only
the retrieved, reranked chunks (bounded by an excerpt/context budget) and is
instructed to cite sources inline with `[n]` markers. Citations are mapped from
the model output back onto the retrieved chunks into structured `Citation`
objects (chunk, document, excerpt, metadata), which the `/query` API returns
alongside the answer.

### 2. Structured logging and request middleware

`app/core/logging.py` emits structured logs (JSON or text) for the root and
uvicorn loggers. `app/core/middleware.py` logs every completed request with
method, path, status, duration, and client. `LOG_LEVEL` and `LOG_FORMAT` are
configurable.

### 3. Health and readiness probes

`/api/v1/health/live` answers whenever the process is up. `/api/v1/health/ready`
runs every dependency probe (database, required API keys) and returns 200/503
with per-dependency results, so orchestrators can route traffic only to healthy
instances.

### 4. Evaluation harness

Retrieval quality is measured with hit rate and MRR against a golden query set
(`data/evaluation/golden_queries.json`). Generation quality is measured with
faithfulness, answer relevancy, and context precision/relevancy. The harness
runs through `scripts/evaluate.py` and writes a JSON report to
`data/evaluation/report.json`.

## Consequences

- **Positive:** answers are verifiable against sources; operations can see why a
  service is failing and what request traffic looks like; quality regressions
  are measurable and attributable to retrieval vs. generation changes.
- **Negative:** grounded generation constrains the prompt and answer shape;
  evaluation depends on carefully curated golden data; structured logging and
  probes add operational surface area.
- **Risk:** golden queries can go stale as documentation changes — mitigated by
  documenting the dataset format and regeneration steps in
  `data/evaluation/README.md` and `docs/evaluation.md`.