from typing import Protocol


class LLMProvider(Protocol):
    """Generate free-text completions from an instruction prompt.

    Providers wrap a concrete language-model backend (OpenAI chat completions,
    hosted models, on-device checkpoints) behind a single ``generate`` call
    used by the grounded-answer use case.
    """

    @property
    def model(self) -> str:
        """Return the identifier of the model backing this provider."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        """Return a completion for ``user_prompt`` guided by ``system_prompt``."""