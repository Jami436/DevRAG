"""Structured logging configuration.

Application logs are emitted to stdout either as JSON (default, machine
parseable) or plain text, controlled by the ``LOG_LEVEL`` and ``LOG_FORMAT``
settings. Callers attach arbitrary structured fields to a record through the
stdlib ``extra={...}`` kwarg; ``JsonFormatter`` lifts them to top-level keys so
tools like Loki, Datadog or Elasticsearch can index them directly.

Wire-up happens once in ``app.main`` via :func:`setup_logging`. Run the service
with ``python -m app.main`` so uvicorn's own logging bootstrap (which would
otherwise replace this configuration) is disabled.
"""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import UTC, datetime
from typing import Any, Literal

from app.core.config import settings as app_settings
from app.core.settings import Settings

#: Standard attributes every :class:`logging.LogRecord` carries. They are
#: projected onto the JSON object under friendlier keys; any remaining keys on
#: the record (normally set through ``extra={...}``) are included verbatim.
_RESERVED_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
    }
)


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    The envelope always contains ``ts`` (UTC, ISO-8601), ``level``, ``logger``
    and ``message`` plus source location fields. Anything passed via
    ``extra={...}`` appears as an additional top-level key, and tracebacks are
    captured under ``exception`` when ``exc_info`` is set.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(
                record.created, tz=UTC
            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        for key, value in vars(record).items():
            if key not in _RESERVED_KEYS:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def build_logging_config(
    *,
    level: str = "INFO",
    log_format: Literal["json", "text"] = "json",
) -> dict[str, Any]:
    """Build a ``logging.config.dictConfig`` structure for the whole service.

    The root logger and every uvicorn logger (startup, error and access) share
    the same console handler so all output goes to ``stdout`` with one coherent
    format. ``disable_existing_loggers`` stays ``False`` so third-party loggers
    (sqlalchemy, httpx, ...) keep working and simply inherit the root handler.
    """
    formatter = "json" if log_format == "json" else "text"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"()": JsonFormatter},
            "text": {"format": _TEXT_FORMAT},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": formatter,
            },
        },
        "root": {"handlers": ["console"], "level": level},
        "loggers": {
            "uvicorn": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
        },
    }


def setup_logging(
    app_settings: Settings = app_settings,
) -> logging.Logger:
    """Configure logging across the process and return the root logger.

    Idempotent: calling it several times reapplies the same structure without
    duplicating handlers. Passes through the application settings so log level
    and format stay consistent with the rest of the configuration.
    """
    config = build_logging_config(
        level=app_settings.log_level,
        log_format=app_settings.log_format,
    )
    logging.config.dictConfig(config)
    return logging.getLogger()