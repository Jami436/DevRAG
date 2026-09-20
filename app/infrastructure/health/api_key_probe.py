from app.domain.health.entities import DependencyStatus


class ApiKeyProbe:
    """Report whether a remote provider's API key is configured.

    Key presence is treated as availability; actually calling the provider from
    a readiness probe would be slow, rate-limited, and noisy.
    """

    def __init__(self, name: str, api_key: str | None) -> None:
        self._name = name
        self._api_key = api_key

    def check(self) -> DependencyStatus:
        if self._api_key:
            return DependencyStatus(name=self._name, state="ok")
        return DependencyStatus(
            name=self._name,
            state="error",
            detail="API key is not configured",
        )