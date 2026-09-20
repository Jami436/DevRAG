import re
from dataclasses import dataclass

from app.domain.generation.entities import Citation
from app.domain.retrieval.entities import RetrievedChunk

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

_SYSTEM_PROMPT = (
    "You are an expert assistant for technical and developer documentation. "
    "Answer the user's question using ONLY the documentation excerpts provided "
    "and never information from outside them. Cite the sources you rely on "
    "inline with their bracketed numbers, e.g. [1] or [2, 3]. If the excerpts "
    "do not contain the answer, say so explicitly and do not invent information."
)


@dataclass(slots=True)
class SourcePassage:
    """A retrieved chunk paired with the excerpt shown to the language model."""

    chunk: RetrievedChunk
    excerpt: str


def system_prompt() -> str:
    """Return the grounding instructions sent to the language model."""
    return _SYSTEM_PROMPT


def build_sources(
    chunks: list[RetrievedChunk],
    *,
    max_excerpt_chars: int = 1000,
    max_total_chars: int = 12000,
) -> tuple[str, list[SourcePassage]]:
    """Render retrieved chunks as numbered documentation for a generation prompt.

    Chunks are numbered ``[1]``..``[N]`` in the order they are given, each
    prefixed with its document title and page number. Passages are truncated to
    ``max_excerpt_chars`` and only as many sources as fit within
    ``max_total_chars`` are included, so the citation indices in the rendered
    text always line up with the passages returned alongside it.
    """
    if max_excerpt_chars <= 0:
        raise ValueError("max_excerpt_chars must be a positive integer")
    if max_total_chars <= 0:
        raise ValueError("max_total_chars must be a positive integer")
    if not chunks:
        return "", []
    passages: list[SourcePassage] = []
    blocks: list[str] = []
    total = 0
    for index, chunk in enumerate(chunks, start=1):
        excerpt = _truncate(chunk.content, max_excerpt_chars)
        block = f"{_source_header(chunk, index)}\n{excerpt}"
        cost = len(block)
        if passages and total + cost > max_total_chars:
            break
        passages.append(SourcePassage(chunk=chunk, excerpt=excerpt))
        blocks.append(block)
        total += cost
    return "\n\n".join(blocks), passages


def user_prompt(question: str, sources: str) -> str:
    """Assemble the question and numbered documentation into a user prompt."""
    if not sources:
        return (
            f"Question: {question}\n\n(No relevant documentation was found. "
            "State that you cannot answer from the provided sources.)"
        )
    return f"Question: {question}\n\nDocumentation:\n{sources}"


def extract_citations(
    answer: str, passages: list[SourcePassage]
) -> list[Citation]:
    """Map the ``[n]`` references in ``answer`` onto the cited chunks.

    Numbers without a matching passage are ignored and duplicates are collapsed,
    preserving the order in which each source is first cited in the answer.
    """
    by_number = {
        number: passage for number, passage in enumerate(passages, start=1)
    }
    citations: list[Citation] = []
    cited: set[int] = set()
    for raw in _CITATION_PATTERN.findall(answer):
        number = int(raw)
        if number not in by_number or number in cited:
            continue
        cited.add(number)
        passage = by_number[number]
        chunk = passage.chunk
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                document_title=chunk.document_title,
                excerpt=passage.excerpt,
                metadata=dict(chunk.metadata),
            )
        )
    return citations


def _source_header(chunk: RetrievedChunk, index: int) -> str:
    label = chunk.document_title or "Untitled document"
    if chunk.page_number > 0:
        return f"[{index}] {label}, page {chunk.page_number}"
    return f"[{index}] {label}"


def _truncate(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + "..."