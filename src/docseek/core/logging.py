"""Structured logging with structlog.

Logs go to stderr (stdout stays free for CLI output) as JSON lines by default. Values of
keys that look like secrets are masked before rendering, and ``SecretStr`` values are
always masked by pydantic itself.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any, TextIO

import structlog

_SECRET_KEY_RE = re.compile(r"(api[_-]?key|token|password|passwd|secret|authorization)", re.I)
REDACTED = "***"


def redact_secrets(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor that masks values of secret-looking keys, recursively."""
    return _redact(event_dict)


def _redact(mapping: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in mapping.items():
        if _SECRET_KEY_RE.search(str(key)):
            mapping[key] = REDACTED
        elif isinstance(value, MutableMapping):
            mapping[key] = _redact(dict(value))
    return mapping


def configure_logging(
    level: str = "INFO", *, json: bool = True, stream: TextIO | None = None
) -> None:
    """Configure structlog for the process. Safe to call more than once."""
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_secrets,
            structlog.processors.dict_tracebacks,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level]),
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Return a logger bound with ``logger=<name>``."""
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger().bind(logger=name)
    return logger
