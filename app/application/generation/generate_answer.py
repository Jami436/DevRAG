from app.domain.generation.citations import (
    build_sources,
    extract_citations,
    system_prompt,
    user_prompt,
)
from app.domain.generation.entities import GeneratedAnswer
from app.domain.generation.interfaces import LLMProvider
from app.domain.retrieval.entities import RetrievedChunk


class GenerateAnswer:
    """Produce a grounded answer with source citations for a question.

    Retrieved chunks are rendered as numbered documentation passages, sent to the
    injected ``LLMProvider`` with strict grounding instructions, and the model's
    ``[n]`` references are mapped back onto the chunks as structured citations.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_excerpt_chars: int = 1000,
        max_total_chars: int = 12000,
    ) -> None:
        if max_excerpt_chars <= 0:
            raise ValueError("max_excerpt_chars must be a positive integer")
        if max_total_chars <= 0:
            raise ValueError("max_total_chars must be a positive integer")
        self._llm_provider = llm_provider
        self._max_excerpt_chars = max_excerpt_chars
        self._max_total_chars = max_total_chars

    def execute(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> GeneratedAnswer:
        sources, passages = build_sources(
            chunks,
            max_excerpt_chars=self._max_excerpt_chars,
            max_total_chars=self._max_total_chars,
        )
        prompt = user_prompt(query, sources)
        answer = self._llm_provider.generate(system_prompt(), prompt)
        citations = extract_citations(answer, passages)
        return GeneratedAnswer(
            query=query,
            answer=answer,
            citations=citations,
            model=self._llm_provider.model,
        )