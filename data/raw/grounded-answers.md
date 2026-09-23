# Grounded Answers and Citations

The final stage of the RAG pipeline turns retrieved chunks into an answer that
is grounded in the documentation and shows its sources.

## Building the generation context

The retrieved chunks are numbered `[1]` to `[N]` in the order they were reranked
and rendered as documentation passages, each prefixed with its document title
and page number. Passages are truncated to a per-source character budget and
only as many sources as fit within the total context budget are included, so the
citation indices in the rendered text always line up with the passages the model
actually saw.

## The grounding instruction

The language model is told to answer the user's question using only the provided
documentation excerpts and never information from outside them, and to cite the
sources it relies on inline with their bracketed numbers, for example `[1]` or
`[2, 3]`. If the excerpts do not contain the answer, the model is instructed to
say so explicitly instead of inventing information.

## Mapping citations back to sources

After generation, the `[n]` references in the model output are parsed and mapped
back to the retrieved chunks. Each citation carries the chunk id, document id,
chunk index, page number, document title and the exact excerpt that grounded the
answer, so consumers can display precisely what the answer was based on. Numbers
without a matching passage are ignored and duplicate references are collapsed,
preserving the order in which each source is first cited.