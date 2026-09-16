from collections.abc import Iterator

from app.domain.documents.chunks import DocumentChunk
from app.domain.documents.entities import Document, DocumentPage


class TokenChunker:
    """Split document pages into fixed-size token windows with configurable overlap.

    A chunk never spans page boundaries: every page is windowed independently and
    each resulting window carries that page's number and metadata.
    """

    def __init__(self, token_limit: int, token_overlap: int = 0) -> None:
        if token_limit <= 0:
            raise ValueError("token_limit must be a positive integer")
        if token_overlap < 0:
            raise ValueError("token_overlap cannot be negative")
        if token_overlap >= token_limit:
            raise ValueError("token_overlap must be smaller than token_limit")
        self._token_limit = token_limit
        self._token_overlap = token_overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        chunk_index = 0
        for page in document.pages:
            for content in self._windows(page.content):
                chunks.append(self._new_chunk(document, page, content, chunk_index))
                chunk_index += 1
        return chunks

    def _windows(self, content: str) -> Iterator[str]:
        tokens = content.split()
        start = 0
        token_count = len(tokens)
        while start < token_count:
            end = min(start + self._token_limit, token_count)
            yield " ".join(tokens[start:end])
            if end == token_count:
                break
            start += self._token_limit - self._token_overlap

    @staticmethod
    def _new_chunk(
        document: Document,
        page: DocumentPage,
        content: str,
        chunk_index: int,
    ) -> DocumentChunk:
        return DocumentChunk(
            document_id=document.id,
            content=content,
            chunk_index=chunk_index,
            page_number=page.page_number,
            metadata=dict(page.metadata),
        )
