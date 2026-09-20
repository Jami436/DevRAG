from functools import lru_cache

from fastapi import APIRouter, Depends, Response

from app.core.config import settings
from app.domain.health.entities import HealthReport
from app.domain.health.interfaces import HealthProbe
from app.infrastructure.health.factory import build_default_health_probes

router = APIRouter()


@lru_cache
def get_health_probes() -> list[HealthProbe]:
    """Build the default dependency probes once per process."""
    return build_default_health_probes(settings)


@router.get("/version", tags=["Health"])
def version_info() -> dict[str, str]:
    """Report the running service name and version."""
    return {"service": settings.app_name, "version": settings.app_version}


@router.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Liveness summary kept for backward compatibility."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/health/live", tags=["Health"])
def liveness_check() -> dict[str, str]:
    """Liveness probe: answers 200 whenever the process is up."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/health/ready", tags=["Health"])
def readiness_check(
    response: Response,
    probes: list[HealthProbe] = Depends(get_health_probes),
) -> HealthReport:
    """Readiness probe: verifies every dependency before serving traffic.

    Returns 200 when all probes report ``ok`` and 503 otherwise, including the
    per-dependency results so operators can tell what is failing.
    """
    checks = [probe.check() for probe in probes]
    healthy = all(check.state == "ok" for check in checks)
    response.status_code = 200 if healthy else 503
    return HealthReport(
        status="ready" if healthy else "unhealthy",
        service=settings.app_name,
        version=settings.app_version,
        dependencies=checks,
    )