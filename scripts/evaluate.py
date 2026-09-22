"""Run DevRAG evaluation suite and write a JSON report.

Evaluates both retrieval quality (hit rate, MRR) and generation quality
(faithfulness, answer relevancy, context precision/relevancy) against a
golden-query set.

Usage:
    python scripts/evaluate.py [--golden-json PATH] [--report PATH]
        [--top-k N] [--skip-generation]

Options:

    --golden-json PATH      Path to golden queries JSON (default: settings
                            ``evaluation_golden_queries_path``).
    --report PATH           Path to write the JSON report (default:
                            ``data/evaluation/report.json``).
    --top-k INTEGER         Number of top chunks to consider per query
                            (default: ``retriever_top_k`` from settings).
    --skip-generation       Skip generation-level metrics (faithfulness,
                            answer relevancy). Only retrieval metrics
                            (hit rate, MRR) will be computed.
"""
import json
import sys
from pathlib import Path

from app.core.settings import settings
from app.infrastructure.evaluation.json_golden_queries_loader import (
    JSONGoldenQueriesLoader,
)
from app.application.evaluation.evaluate_retrieval import EvaluateRetrieval
from app.application.evaluation.evaluate_generation import EvaluateGeneration
from app.infrastructure.llm.factory import build_llm_provider
from app.services.search import build_default_search_service


if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
    print(__doc__)
    sys.exit(0)

argv = sys.argv[1:]


def _build_retriever(top_k: int):
    """Return a ``query -> list[RetrievedChunk]`` callable wired from settings."""
    search_service = build_default_search_service(settings)
    return lambda query: search_service.execute(query, top_k=top_k)


def _build_generator():
    """Return a ``(query, chunks) -> GeneratedAnswer`` callable, or None.

    The generator is built only when an LLM provider is configured; otherwise
    ``None`` is returned and the caller can skip generation-level metrics.
    """
    try:
        llm_provider = build_llm_provider(settings)
        from app.application.generation.generate_answer import GenerateAnswer

        generator = GenerateAnswer(
            llm_provider=llm_provider,
            max_excerpt_chars=settings.generation_max_excerpt_chars,
            max_total_chars=settings.generation_max_context_chars,
        )

        def _generator(query: str, chunks: list) -> "GeneratedAnswer":
            return generator.execute(query, chunks)

        return _generator
    except Exception as exc:  # pragma: no cover - runtime configuration error
        print(
            f"WARNING: could not build generation provider: {exc!r}. "
            "Skipping generation-level metrics.",
            file=sys.stderr,
        )
        return None


def main(argv: list[str] | None = None) -> None:
    """Entry point for the evaluation runner."""
    # Parse simple CLI arguments
    gold_path = settings.evaluation_golden_queries_path
    report_path = "data/evaluation/report.json"
    top_k = settings.retriever_top_k
    skip_gen = False

    i = 0
    while i < len(argv or []):
        arg = argv[i]
        if arg == "--golden-json" and i + 1 < len(argv or []):
            gold_path = argv[i + 1]
            i += 2
        elif arg == "--report" and i + 1 < len(argv or []):
            report_path = argv[i + 1]
            i += 2
        elif arg == "--top-k" and i + 1 < len(argv or []):
            top_k = int(argv[i + 1])
            i += 2
        elif arg == "--skip-generation":
            skip_gen = True
            i += 1
        else:
            i += 1

    # Load golden queries
    loader = JSONGoldenQueriesLoader(gold_path)
    try:
        queries = loader.load()
    except FileNotFoundError as exc:
        print(f"ERROR: golden query file not found: {exc}", file=sys.stderr)
        sys.exit(1)

    # Build retriever
    retriever = _build_retriever(top_k)

    # Optionally build generator
    generator = None
    if not skip_gen:
        generator = _build_generator()

    # Run retrieval evaluation
    from app.application.evaluation.evaluate_retrieval import EvaluateRetrieval
    retrieval_use_case = EvaluateRetrieval(loader, retriever, top_k=top_k)
    retrieval_report = retrieval_use_case.execute()

    # Run generation evaluation (if generator available)
    generation_report = None
    if generator is not None:
        from app.application.evaluation.evaluate_generation import (
            EvaluateGeneration,
        )
        generation_use_case = EvaluateGeneration(
            loader, retriever, generator, top_k=top_k
        )
        generation_report = generation_use_case.execute()

    # Assemble output report
    report = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "top_k": top_k,
        "golden_queries_path": gold_path,
        "retrieval": {
            "num_queries": retrieval_report.num_queries,
            "hit_rate": retrieval_report.hit_rate,
            "mean_reciprocal_rank": retrieval_report.mean_reciprocal_rank,
            "per_query": [
                {
                    "query": eq.query,
                    "hit": eq.hit,
                    "reciprocal_rank": eq.reciprocal_rank,
                    "retrieved_chunk_ids": [
                        str(cid) for cid in eq.retrieved_chunk_ids
                    ],
                    "relevant_chunk_ids": [
                        str(cid) for cid in eq.relevant_chunk_ids
                    ],
                }
                for eq in retrieval_report.per_query
            ],
        },
    }

    if generation_report is not None:
        report["generation"] = {
            "num_answers": generation_report.num_queries,
            "mean_faithfulness": generation_report.mean_faithfulness,
            "mean_answer_relevancy": generation_report.mean_answer_relevancy,
            "mean_context_precision": generation_report.mean_context_precision,
            "mean_context_relevancy": generation_report.mean_context_relevancy,
            "per_answer": [
                {
                    "query": ea.query,
                    "answer": ea.answer,
                    "faithfulness": ea.faithfulness,
                    "answer_relevancy": ea.answer_relevancy,
                    "context_precision": ea.context_precision,
                    "context_relevancy": ea.context_relevancy,
                    "top_k": ea.top_k,
                }
                for ea in generation_report.per_answer
            ],
        }

    # Write report
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2, default=str))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main(sys.argv[1:])