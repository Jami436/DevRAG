import re

_SOURCE_REFERENCE = re.compile(r"\[(\d+)\]")

_NO_DOCUMENTATION_ANSWER = (
    "The provided documentation does not contain enough information to "
    "answer this question."
)


class CannedLLMProvider:
    """Deterministic placeholder generator for offline development.

    Emits a reply that cites every numbered source present in the prompt, so the
    grounded-answer pipeline and citation extraction can be exercised end to end
    without a paid model or API key.
    """

    @property
    def model(self) -> str:
        return "devrag-canned"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        del system_prompt, max_tokens
        marker = user_prompt.find("Documentation:")
        if marker == -1:
            return _NO_DOCUMENTATION_ANSWER
        numbers = _SOURCE_REFERENCE.findall(user_prompt[marker:])
        if not numbers:
            return _NO_DOCUMENTATION_ANSWER
        refs = ", ".join(f"[{number}]" for number in numbers)
        return (
            "Based on the documentation provided, the answer to this question is "
            f"supported by the following sources: {refs}."
        )