from openai import OpenAI


class OpenAILLMProvider:
    """Generate text completions through the OpenAI Chat Completions API.

    ``max_tokens`` configured at construction bounds every request unless a
    per-call override is supplied. The concrete client is created eagerly so a
    missing API key fails fast at build time rather than on first request.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("An OpenAI API key is required")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self._model = model
        self._max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        token_budget = max_tokens or self._max_tokens
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=token_budget,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""