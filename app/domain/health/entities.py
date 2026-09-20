from dataclasses import dataclass, field
from typing import Literal

DependencyState = Literal["ok", "error"]


@dataclass(slots=True)
class DependencyStatus:
    """Outcome of probing a single external dependency."""

    name: str
    state: DependencyState
    detail: str | None = None


@dataclass(slots=True)
class HealthReport:
    """Aggregate readiness of all probed dependencies."""

    status: Literal["ready", "unhealthy"]
    service: str
    version: str
    dependencies: list[DependencyStatus] = field(default_factory=list)