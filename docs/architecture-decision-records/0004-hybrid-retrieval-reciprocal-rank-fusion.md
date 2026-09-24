# ADR-0004: Hybrid retrieval with reciprocal rank fusion and reranking

- **Status:** Accepted
- **Date:** 2026-09-24

## Context

Technical documentation often contains exact keyword matches (APIs, error
signals, identifiers) that pure vector search misses, and semantic matches that
pure keyword search misses. A single retrieval "signal" underperforms on
developer questions. Greedy top-k selection from one list also ignores how well
a chunk ranked under the *other* signal.

## Decision

Implement **hybrid retrieval**:

1. Run **vector** similarity search and **keyword** (Postgres full-text) search
   in parallel over the same chunk set.
2. Combine the two ranked lists with **Reciprocal Rank Fusion (RRF)** —
   `sum(1 / (k + rank))` — producing a single fused ordering that is robust to
   score-scale differences between signals (`app/domain/retrieval/fusion.py`).
3. Apply a **reranking** layer on the fused candidates: a cross-encoder,
   an LLM-based reranker, or an identity pass-through, selected by
   `RERANKER_PROVIDER`. The top `top_k` chunks are returned with their
   per-signal and final scores.

Fusion preference (vector vs. keyword weight), the fusion constant `k`, the
hybrid candidate pool, and `top_k` are all configurable in settings.

## Consequences

- **Positive:** better recall on mixed semantic/keyword queries; RRF avoids
  score-normalization problems; reranking re-scores explicitly by query–chunk
  relevance, improving precision at `top_k`.
- **Negative:** two queries per retrieval (cost and latency); reranking adds an
  extra model pass.
- **Risk:** reranker availability (network/API costs) — mitigated by the
  identity reranker for offline use and by making the layer optional.