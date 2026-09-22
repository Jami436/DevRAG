"""HTTP middleware for structured request logging."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import Response

access_logger = logging.getLogger("app.access")


async def request_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Emit one structured record per completed HTTP request.

    The record carries method, path, response status, wall-clock duration in
    milliseconds and the connecting client host, so request traffic can be
    sliced and correlated from the JSON log stream.
    """
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    access_logger.info(
        "request completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "client": request.client.host if request.client is not None else None,
        },
    )
    return response