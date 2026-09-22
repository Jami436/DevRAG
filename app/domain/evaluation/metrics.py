from collections.abc import Collection, Sequence
from uuid import UUID

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "with",
        "this",
        "that",
        "from",
        "your",
        "you",
        "our",
        "not",
        "can",
        "why",
        "what",
        "how",
        "does",
        "will",
        "have",
        "has",
        "was",
        "its",
        "all",
        "any",
        "but",
        "or",
        "if",
        "of",
        "in",
        "on",
        "at",
        "to",
        "a",
        "an",
        "is",
        "be",
        "by",
        "as",
        "it",
    }
)


def _content_tokens(text: str) -> frozenset[str]:
    """Return the set of content-bearing tokens from *text*.

    Tokens are lowercased alphabetic words with length >= 3 that are not
    in the project's stopword set. These tokens represent the semantic
    "substance" of a piece of text for overlap-based metrics.
    """
    tokens: set[str] = set()
    for word in text.split():
        w = word.lower().strip(".,;:!?'\"")
        if len(w) >= 3 and w.isalpha() and w not in _STOPWORDS:
            tokens.add(w)
    return frozenset(tokens)


def is_hit(
    retrieved_chunk_ids: Sequence[UUID],
    relevant_chunk_ids: Collection[UUID],
) -> bool:
    """Return True when any retrieved chunk is in the relevant set."""
    return any(chunk_id in relevant_chunk_ids for chunk_id in retrieved_chunk_ids)


def reciprocal_rank(
    retrieved_chunk_ids: Sequence[UUID],
    relevant_chunk_ids: Collection[UUID],
) -> float:
    """Return the reciprocal of the first relevant chunk's rank, or 0.

    Ranks are 1-based. A chunk at position ``n`` contributes ``1 / n``; when no
    retrieved chunk is relevant the contribution is zero.
    """
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0


def faithfulness(answer: str, context: str) -> float:
    """Fraction of the answer's content tokens that also appear in the context.

    Faithfulness measures whether the generated answer is grounded in the
    retrieved context. A higher score indicates more of the answer's
    substantive content is supported by the context.

    Returns a value in [0, 1]; returns 1.0 when the answer contains no
    content tokens (e.g. a pure disclaimer), and 0.0 when none of the
    answer's content tokens appear in the context.
    """
    answer_tokens = _content_tokens(answer)
    context_tokens = _content_tokens(context)
    if not answer_tokens:
        return 1.0
    matched = sum(1 for tok in answer_tokens if tok in context_tokens)
    return matched / len(answer_tokens)


def answer_relevancy(query: str, answer: str) -> float:
    """Fraction of the query's content tokens that appear in the answer.

    Answer relevancy measures whether the generated answer directly addresses
    the user's question. A higher score indicates more of the query's
    substantive terms are covered by the answer.

    Returns a value in [0, 1]; returns 1.0 when the query has no content
    tokens, and 0.0 when none of the query's content tokens appear in the
    answer.
    """
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return 1.0
    answer_tokens = _content_tokens(answer)
    matched = sum(1 for tok in query_tokens if tok in answer_tokens)
    return matched / len(query_tokens)


def context_precision(
    retrieved_chunk_ids: Sequence[UUID],
    relevant_chunk_ids: Collection[UUID],
) -> float:
    """Rank-aware precision (mean average precision) of retrieved chunks.

    For each relevant chunk that appears at position *k* in the retrieved
    list, the precision at that rank (*k*) is accumulated. The sum is
    divided by the total number of relevant chunks so a perfect ranking
    scores 1.0 and no relevant chunks in the retrieval scores 0.0.
    """
    relevant = set(relevant_chunk_ids)
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def context_relevancy(
    retrieved_chunk_ids: Sequence[UUID],
    relevant_chunk_ids: Collection[UUID],
) -> float:
    """Fraction of retrieved chunks that are relevant to the query.

    Context relevancy measures the signal-to-noise ratio of the retrieved
    context: what proportion of the chunks returned by the retriever are
    actually relevant to the user's question.
    """
    if not retrieved_chunk_ids:
        return 0.0
    relevant = set(relevant_chunk_ids)
    hits = sum(1 for chunk_id in retrieved_chunk_ids if chunk_id in relevant)
    return hits / len(retrieved_chunk_ids)