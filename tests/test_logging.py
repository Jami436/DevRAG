import json
import logging
from collections.abc import Iterator

import pytest

from app.core.config import settings
from app.core.logging import JsonFormatter, build_logging_config, setup_logging


class _NonSerializableObject:
    def __str__(self) -> str:
        return "nonserializable"


def _record(
    level: int = logging.INFO,
    *,
    exc_info: tuple[type[BaseException], BaseException, None] | None = None,
    extra: dict[str, object] | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.test",
        level=level,
        pathname=__file__,
        lineno=42,
        msg="hello %s",
        args=("world",),
        exc_info=exc_info,
    )
    record.funcName = "some_function"
    if extra:
        for key, value in extra.items():
            setattr(record, key, value)
    return record


def test_json_formatter_projects_structured_fields() -> None:
    payload = json.loads(
        JsonFormatter().format(
            _record(extra={"document_id": 7, "query": "knn"})
        )
    )

    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["message"] == "hello world"
    assert payload["module"] == "test_logging"
    assert payload["function"] == "some_function"
    assert payload["line"] == 42
    assert payload["document_id"] == 7
    assert payload["query"] == "knn"


def test_json_formatter_hides_reserved_record_attributes() -> None:
    payload = json.loads(JsonFormatter().format(_record()))

    for reserved in ("msg", "args", "exc_info", "created", "thread"):
        assert reserved not in payload


def test_json_formatter_serializes_unknown_values() -> None:
    payload = json.loads(
        JsonFormatter().format(_record(extra={"obj": _NonSerializableObject()}))
    )

    assert payload["obj"] == "nonserializable"


def test_json_formatter_includes_exception_traceback() -> None:
    exc_info = (ValueError, ValueError("boom"), None)
    payload = json.loads(JsonFormatter().format(_record(exc_info=exc_info)))

    assert "ValueError" in payload["exception"]


def test_build_logging_config_defaults_to_json() -> None:
    config = build_logging_config()

    assert config["version"] == 1
    assert config["disable_existing_loggers"] is False
    assert config["root"] == {"handlers": ["console"], "level": "INFO"}
    assert config["handlers"]["console"]["formatter"] == "json"
    assert config["formatters"]["json"]["()"] is JsonFormatter
    assert config["loggers"]["uvicorn.access"]["propagate"] is False


def test_build_logging_config_supports_text_format() -> None:
    config = build_logging_config(level="DEBUG", log_format="text")

    assert config["root"]["level"] == "DEBUG"
    assert config["handlers"]["console"]["formatter"] == "text"


@pytest.fixture()
def restore_logging() -> Iterator[None]:
    """Restore the logging graph mutated by ``setup_logging``."""
    root = logging.getLogger()
    previous_level = root.level
    previous_handlers = list(root.handlers)

    uvicorn_loggers = {
        name: logging.getLogger(name)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access")
    }
    previous_handlers_by_logger = {
        name: list(logger.handlers) for name, logger in uvicorn_loggers.items()
    }

    yield

    root.setLevel(previous_level)
    root.handlers[:] = previous_handlers
    for name, logger in uvicorn_loggers.items():
        logger.handlers[:] = previous_handlers_by_logger[name]


def test_setup_logging_applies_settings(restore_logging: None) -> None:
    logging.getLogger().setLevel(logging.CRITICAL)

    result = setup_logging(settings)

    assert result is logging.getLogger()
    assert result.level == getattr(logging, settings.log_level.upper())

    json_handlers = [
        handler
        for handler in result.handlers
        if isinstance(handler.formatter, JsonFormatter)
    ]
    assert json_handlers
    assert logging.getLogger("uvicorn.access").propagate is False


def test_setup_logging_emits_json_lines(restore_logging: None, capsys) -> None:
    setup_logging(settings)

    logging.getLogger("app.test").info(
        "request completed", extra={"method": "GET", "status": 200}
    )

    line = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(line)

    assert record["level"] == "INFO"
    assert record["message"] == "request completed"
    assert record["method"] == "GET"
    assert record["status"] == 200