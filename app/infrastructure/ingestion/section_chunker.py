import re
from collections.abc import Iterator

from app.domain.documents.chunks import DocumentChunk
from app.domain.documents.entities import Document, DocumentPage

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


class SectionChunker:
    """Split document pages into chunks while keeping sections intact.

    Sections are the unit of meaning: whole pages are grouped into a single chunk
    while they fit the token budget, so a page is only ever broken when it exceeds
    the budget alone. Oversized pages are first split on blank-line paragraph
    boundaries, falling back to token windows only for paragraphs that are still
    too large. Every sub-chunk of a split page is re-prefixed with the section
    heading so each chunk stays semantically self-contained.
    """

    def __init__(self, token_limit: int) -> None:
        if token_limit <= 0:
            raise ValueError("token_limit must be a positive integer")
        self._token_limit = token_limit

    def chunk(self, document: Document) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        chunk_index = 0
        pending: list[str] = []
        pending_tokens = 0
        pending_page: DocumentPage | None = None

        def flush() -> None:
            nonlocal chunk_index, pending, pending_tokens, pending_page
            if pending_page is None:
                return
            content = "\n\n".join(pending)
            chunks.append(self._new_chunk(document, pending_page, content, chunk_index))
            chunk_index += 1
            pending = []
            pending_tokens = 0
            pending_page = None

        for page in document.pages:
            page_tokens = len(page.content.split())
            if page_tokens == 0:
                continue

            if page_tokens > self._token_limit:
                flush()
                for content in self._split_large(page):
                    chunks.append(self._new_chunk(document, page, content, chunk_index))
                    chunk_index += 1
                continue

            if (
                pending_page is not None
                and pending_tokens + page_tokens > self._token_limit
            ):
                flush()

            pending.append(page.content)
            pending_tokens += page_tokens
            if pending_page is None:
                pending_page = page

        flush()

        return chunks

    def _split_large(self, page: DocumentPage) -> list[str]:
        paragraphs = [
            paragraph.strip()
            for paragraph in _PARAGRAPH_SPLIT.split(page.content)
            if paragraph.strip()
        ]
        heading = page.metadata.get("heading", "")

        parts: list[str] = []
        pending: list[str] = []
        pending_tokens = 0

        def flush() -> None:
            nonlocal pending, pending_tokens
            if pending:
                parts.append("\n\n".join(pending))
                pending = []
                pending_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = len(paragraph.split())
            if paragraph_tokens > self._token_limit:
                flush()
                parts.extend(self._windows(paragraph))
                continue
            if pending and pending_tokens + paragraph_tokens > self._token_limit:
                flush()
            pending.append(paragraph)
            pending_tokens += paragraph_tokens

        flush()

        for index, part in enumerate(parts):
            if index > 0:
                parts[index] = self._prefix_heading(part, heading)
        return parts

    def _windows(self, content: str) -> Iterator[str]:
        tokens = content.split()
        start = 0
        token_count = len(tokens)
        while start < token_count:
            end = min(start + self._token_limit, token_count)
            yield " ".join(tokens[start:end])
            if end == token_count:
                break
            start += self._token_limit

    @staticmethod
    def _prefix_heading(content: str, heading: str) -> str:
        if not heading or content.startswith(heading):
            return content
        return f"{heading}\n{content}"

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
