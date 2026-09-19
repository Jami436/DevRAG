import json

from openai import OpenAI

from app.domain.retrieval.entities import RetrievedChunk

_MAX_EXCERPT_CHARS = 1000


class LLMReranker:
    """Re-rank chunks by asking an LLM to score candidate relevance.

    Candidates are sent to chat completions as excerpts with a request for a
    strict JSON score card (0-10 per chunk). Scores are parsed back, chunks are
    re-ordered by descending score, and any chunk the model omitted is treated
    as a zero. Content is truncated per passage so prompts stay within token
    budgets.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_excerpt_chars: int = _MAX_EXCERPT_CHARS,
    ) -> None:
        if not api_key:
            raise ValueError("An OpenAI API key is required")
        if max_excerpt_chars <= 0:
            raise ValueError("max_excerpt_chars must be a positive integer")
        self._model = model
        self._max_excerpt_chars = max_excerpt_chars
        self._client = OpenAI(api_key=api_key)

    def rerank(
        self, query: str, results: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not results:
            return []
        prompt = self._build_prompt(query, results)
        response = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        content = response.choices[0].message.content or "{}"
        scores = self._parse_scores(content)
        ordered = sorted(
            results,
            key=lambda hit: scores.get(str(hit.chunk_id), 0.0),
            reverse=True,
        )
        for rank, hit in enumerate(ordered, start=1):
            hit.final_score = float(scores.get(str(hit.chunk_id), 0.0))
            hit.rank = rank
        return ordered

    def _build_prompt(
        self, query: str, results: list[RetrievedChunk]
    ) -> str:
        passages = "\n".join(
            f"[{hit.chunk_id}] {self._excerpt(hit.content)}"
            for hit in results
        )
        return (
            'You are a relevance scorer for a RAG system. Score how useful each '
            'candidate passage is for answering the question with an integer '
            "between 0 and 10. Respond with JSON only, in the exact shape: "
            '{"scores": [{"id": "<chunk id>", "score": 5}]}.\n\n'
            f"Question: {query}\n\nPassages:\n{passages}"
        )

    def _parse_scores(self, content: str) -> dict[str, float]:
        payload = json.loads(content)
        scores: dict[str, float] = {}
        for item in payload.get("scores", []):
            chunk_id = item.get("id")
            score = item.get("score")
            if isinstance(chunk_id, str) and isinstance(score, (int, float)):
                scores[chunk_id] = float(score)
        return scores

    def _excerpt(self, content: str) -> str:
        if len(content) <= self._max_excerpt_chars:
            return content
        return content[: self._max_excerpt_chars] + "..."