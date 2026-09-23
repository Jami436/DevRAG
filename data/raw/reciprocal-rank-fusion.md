# Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) is the algorithm DevRAG uses to merge the dense
vector and keyword full-text rankings into one result list.

## The rank-based score

Instead of normalizing the different score scales produced by each retrieval
signal, RRF scores every chunk from its rank position alone. A chunk ranked at
position `r` in one list contributes `1 / (k + r)` to its fused score, where
`k` is a constant (60 by default) that damps the influence of very high ranks.
Every chunk's contributions are summed across the lists, and chunks are re-sorted
by the total.

## Properties

The formula never relies on the raw similarity values, so it works even when the
two signals use incomparable scales, and it is robust to noisy scores. A chunk
that ranks well in both lists outperforms one that ranks first in a single
list, which is exactly the behaviour a hybrid aggregator should have.

## Configuration

The fusion constant is exposed as `RETRIEVAL_FUSION_K` and the weight applied
to each signal as `RETRIEVAL_HYBRID_VECTOR_WEIGHT` and
`RETRIEVAL_HYBRID_KEYWORD_WEIGHT`. Candidates from each source are fetched with
`RETRIEVAL_HYBRID_CANDIDATES` (wider than the final top-k so the fusion has
enough spanning candidates to merge), and the merged list is truncated back to
the requested limit.