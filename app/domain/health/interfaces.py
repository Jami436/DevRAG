from typing import Protocol

from app.domain.health.entities import DependencyStatus


class HealthProbe(Protocol):
    """Probe a single dependency and report its availability."""

    def check(self) -> DependencyStatus:
        """Return the current state of the dependency."""