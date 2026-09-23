# Hybrid Retrieval

DevRAG answers questions over developer documentation by combining two
independent retrieval signals: dense vector search and keyword full-text
search. Each signal alone has blind spots, so the platform fuses them into a
single ranked list.

## Dense vector search

Every indexed chunk is embedded with a sentence transformer (or an API-based
embedding model) into a high-dimensional vector. A question is embedded the
same way, and PostgreSQL with pgvector returns the chunks whose vectors are
cosine-closest to the query vector. Vector search captures *semantic*
similarity: it can match a question to an answer even when the two share no
words in common, because meaning is encoded in the embeddings.

## Keyword full-text search

In parallel, the query text is run through PostgreSQL full-text search against
the chunk contents using the `english` configuration. Results are ranked with
`ts_rank_cd`, which rewards frequent term matches in short documents. Keyword
search excels at exact terms, identifiers, function names and version strings
that an embedding may blur together.

## Fusing the signals

The two ranked lists are merged with Reciprocal Rank Fusion, which combines the
rank positions of each chunk from both lists instead of trying to normalize
incomparable score scales. The fusion gives strong results even when vector and
keyword scores cannot be compared directly. Chunks that appear high in both
lists surface at the top of the merged result.

## Why hybrid beats either signal alone

Vector search alone can miss rare technical terms that the model never saw
during training. Keyword search alone misses paraphrases and synonyms. Hybrid
search keeps both strengths and compensates for their individual weaknesses,
which makes it the default retrieval strategy on DevRAG.