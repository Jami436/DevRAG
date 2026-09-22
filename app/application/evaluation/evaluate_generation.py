from collections.abc import Callable

from app.domain.evaluation.entities import AnswerEvaluation, GenerationReport
from app.domain.evaluation.interfaces import GoldenQueryLoader
from app.domain.evaluation.metrics import (
    answer_relevancy,
    context_precision,
    context_relevancy,
    faithfulness,
)
from app.domain.generation.entities import GeneratedAnswer
from app.domain.retrieval.entities import RetrievedChunk


class EvaluateGeneration:
    """Evaluate a generative RAG pipeline on faithfulness, answer relevancy,
    context precision, and context relevancy against golden queries.

    For each golden query the retriever returns at most ``top_k`` chunks, a
    generator produces an answer from those chunks, and four metrics are
    computed to assess the quality of the generated answer.
    """

    def __init__(
        self,
        loader: GoldenQueryLoader,
        retriever: Callable[[str], list[RetrievedChunk]],
        generator: Callable[[str, list[RetrievedChunk]], GeneratedAnswer],
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self._loader = loader
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k

    def execute(self) -> GenerationReport:
        queries = self._loader.load()
        if not queries:
            raise ValueError("golden query set must not be empty")

        per_answer: list[AnswerEvaluation] = []
        for golden in queries:
            chunks = self._retriever(golden.query)[: self._top_k]
            answer: GeneratedAnswer = self._generator(golden.query, chunks)

            context = "\n".join(chunk.content for chunk in chunks)
            retrieved_ids = [chunk.chunk_id for chunk in chunks]
            relevant_ids = set(golden.relevant_chunk_ids)

            per_answer.append(
                AnswerEvaluation(
                    query=golden.query,
                    answer=answer.answer,
                    context=context,
                    faithfulness=faithfulness(answer.answer, context),
                    answer_relevancy=answer_relevancy(golden.query, answer.answer),
                    context_precision=context_precision(
                        retrieved_ids, relevant_ids
                    ),
                    context_relevancy=context_relevancy(
                        retrieved_ids, relevant_ids
                    ),
                    top_k=self._top_k,
                )
            )

        num_queries = len(per_answer)
        scores_faithfulness = [
            e.faithfulness for e in per_answer
        ]
        scores_answer_relevancy = [
            e.answer_relevancy for e in per_answer
        ]
        scores_context_precision = [
            e.context_precision for e in per_answer
        ]
        scores_context_relevancy = [
            e.context_relevancy for e in per_answer
        ]

        mean_faithfulness = sum(scores_faithfulness) / num_queries
        mean_answer_relevancy = sum(scores_answer_relevancy) / num_queries
        mean_context_precision = sum(scores_context_precision) / num_queries
        mean_context_relevancy = sum(scores_context_relevancy) / num_queries

        return GenerationReport(
            num_queries=num_queries,
            mean_faithfulness=mean_faithfulness,
            mean_answer_relevancy=mean_answer_relevancy,
            mean_context_precision=mean_context_precision,
            mean_context_relevancy=mean_context_relevancy,
            per_answer=per_answer,
        )