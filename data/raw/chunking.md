# Chunking

Chunking is the step that splits a parsed document into the retrieval units
that get embedded and stored. The quality of chunking directly drives the
quality of retrieval, so DevRAG ships two strategies.

## Section chunking

The default strategy keeps whole documentation sections intact. Pages are
grouped into a single chunk while they fit the token budget, so a section is
only ever broken when it exceeds the budget on its own. Oversized sections are
first split on blank-line paragraph boundaries and fall back to token windows
only for paragraphs that are still too large. Every sub-chunk of a split
section is re-prefixed with the section heading so each chunk stays
semantically self-contained.

## Token chunking

The alternative strategy splits text into fixed-size windows of tokens with a
configurable overlap. `CHUNKING_TOKEN_LIMIT` controls the window size and
`CHUNKING_TOKEN_OVERLAP` controls how many tokens the neighbouring windows
share. Overlap keeps a topic that straddles a boundary from being split out of
both windows.

## How chunking affects retrieval quality

Sections make good chunks because they are natural units of meaning: a question
about a feature is usually answered by one section, and the surrounding text
gives the model the context it needs. Fixed token windows are uniform and
simple, but they can cut a topic in half and leave each half too small to
answer a question. Small chunks are precise but lack context; large chunks are
context-rich but dilute the relevance of any single span. The right balance
depends on the documentation and the questions asked against it.