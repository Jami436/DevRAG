"""Run an end-to-end demo: ingest a real doc set and show a grounded answer.

Walks a directory of documentation files (Markdown, HTML, PDF), ingests every
file through the default parse -> chunk -> embed -> store pipeline, then runs
hybrid retrieval + reranking and generates a grounded answer with source
citations for a set of questions.

Usage:
    python scripts/demo.py [--docs-dir PATH] [--query "question"] [--top-k N]
        [--no-generation]

Options:

    --docs-dir PATH       Directory of documentation files to ingest (default:
                          ``data/raw``).
    --query "QUESTION"    Ask a single question instead of the default set.
    --top-k INTEGER       Number of chunks to retrieve per question (default:
                          ``retriever_top_k`` from settings).
    --no-generation       Only show retrieval; skip the grounded answer.

The demo reuses the exact providers wired by ``app/core/settings.py``, so it
honours ``EMBEDDING_PROVIDER``, ``RERANKER_PROVIDER`` and
``GENERATION_PROVIDER``. It requires a live PostgreSQL with pgvector
(``DATABASE_URL``) and, depending on configuration, an ``OPENAI_API_KEY``
or the on-device sentence-transformers model. Generation needs an LLM provider
(``GENERATION_PROVIDER=openai`` with an API key, or ``none`` for the offline
canned provider); without one the script reports retrieval-only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.settings import settings
from app.services.documents import DocumentService, build_default_document_service
from app.services.query import QueryService, build_default_query_service
from app.services.search import build_default_search_service

_SUPPORTED_EXTENSIONS = (".md", ".markdown", ".html", ".htm", ".pdf")
_PREVIEW_CHARS = 180

_DEFAULT_QUESTIONS = [
    "What is hybrid search and how does it work?",
    "How does chunking affect retrieval quality?",
    "What is reciprocal rank fusion?",
    "How are citations mapped from model output to source chunks?",
]


class _DemoError(RuntimeError):
    """Raised when the demo cannot run with the current configuration."""


def _banner(line: str, width: int = 72) -> str:
    return "\n" + line + "\n" + "-" * width


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit] + "..."


def _score(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "-"


def _collect_documents(docs_dir: Path) -> list[Path]:
    if not docs_dir.is_dir():
        raise _DemoError(f"Docs directory not found: {docs_dir}")
    files = [
        path
        for path in sorted(docs_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS
    ]
    return files


def _ingest(docs_dir: Path, service: DocumentService) -> int:
    files = _collect_documents(docs_dir)
    if not files:
        raise _DemoError(
            f"No supported documents ({', '.join(_SUPPORTED_EXTENSIONS)}) "
            f"found under {docs_dir}"
        )
    items, _total = service.list(limit=1000, offset=0)
    existing = {doc.title for doc in items}
    total_chunks = 0
    print(f"Found {len(files)} document(s) under {docs_dir}")
    for path in files:
        if path.stem in existing:
            print(f"  - skip {path.name} (already indexed as {path.stem!r})")
            continue
        document = service.ingest(path)
        chunks = service.get_chunks(document.id)
        total_chunks += len(chunks)
        print(
            f"  - ingest {path.name}: {len(chunks)} chunk(s), "
            f"document {document.id}"
        )
    return total_chunks


def _build_query_service() -> QueryService | None:
    try:
        return build_default_query_service(settings)
    except ValueError as exc:
        print(
            f"\nWARNING: generation disabled ({exc!r}). "
            "Set GENERATION_PROVIDER=openai with OPENAI_API_KEY, or "
            "GENERATION_PROVIDER=none for the offline canned provider.",
            file=sys.stderr,
        )
        return None


def _answer(
    question: str, top_k: int, query_service: QueryService | None
) -> None:
    print(_banner(f"Q: {question}"))

    if query_service is None:
        print("  (generation unavailable; skipping grounded answer)")
        return

    answer = query_service.execute(question, top_k=top_k)
    print(f"\nAnswer (model={answer.model}):\n")
    print(answer.answer)
    if not answer.citations:
        print("\n  (no citations in the answer)")
        return
    print("\nCitations:")
    for citation in answer.citations:
        title = citation.document_title or "untitled"
        excerpt = _preview(citation.excerpt or citation.metadata.get("heading", ""))
        print(
            f"  [{citation.chunk_index}] {title}  p.{citation.page_number}  "
            f"chunk {citation.chunk_id} — {excerpt}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a doc set and show a grounded answer.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--docs-dir",
        default="data/raw",
        help="Directory of documentation files to ingest (default: data/raw).",
    )
    parser.add_argument(
        "--query",
        help="Ask a single question instead of the default set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.retriever_top_k,
        help="Chunks to retrieve per question (default: settings).",
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Only show retrieval results, skip the grounded answer.",
    )
    args = parser.parse_args(argv)

    print("=" * 72)
    print(" DevRAG demo — ingest a real doc set & show a grounded answer")
    print("=" * 72)
    embedding_model = settings.openai_embedding_model
    if settings.embedding_provider != "openai":
        embedding_model = settings.local_embedding_model
    print(f"Embedding: {settings.embedding_provider} ({embedding_model})")
    print(
        f"Chunking : {settings.chunking_strategy} (token limit "
        f"{settings.chunking_token_limit})"
    )
    print(f"Reranker : {settings.reranker_provider}")
    print(f"LLM      : {settings.generation_provider} ({settings.generation_model})")

    try:
        document_service = build_default_document_service()
    except ValueError as exc:
        print(
            f"\nERROR: could not build the document pipeline: {exc!r}\n"
            "Check EMBEDDING_PROVIDER / OPENAI_API_KEY / DATABASE_URL.",
            file=sys.stderr,
        )
        return 1

    try:
        print(_banner("Ingestion"))
        total_chunks = _ingest(Path(args.docs_dir), document_service)
        print(f"Done: {total_chunks} chunk(s) stored across documents.")
    except _DemoError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    try:
        search_service = build_default_search_service(settings)
    except ValueError as exc:
        print(
            f"\nERROR: could not build the search service: {exc!r}",
            file=sys.stderr,
        )
        return 1

    query_service: QueryService | None = None
    if not args.no_generation:
        query_service = _build_query_service()

    questions = [args.query] if args.query else _DEFAULT_QUESTIONS

    for question in questions:
        print(_banner(f"Retrieval for: {question}"))
        hits = search_service.execute(question, top_k=args.top_k)
        if not hits:
            print("  (no chunks retrieved)")
            continue
        for index, hit in enumerate(hits, start=1):
            title = hit.document_title or "untitled"
            heading = hit.metadata.get("heading", "")
            print(
                f"  [{index}] (rank {hit.rank}, vector {_score(hit.vector_score)}, "
                f"keyword {_score(hit.keyword_score)}, "
                f"fused {_score(hit.final_score)})"
            )
            print(f"       {title}  p.{hit.page_number}  chunk {hit.chunk_index}")
            print(f"       {_preview(hit.content)}")
            if heading:
                print(f"       heading: {heading}")
        _answer(question, args.top_k, query_service)

    print("\nDemo complete. Sample docs live in data/raw; inspect the golden")
    print("queries and scripts/evaluate.py for a full evaluation run.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))