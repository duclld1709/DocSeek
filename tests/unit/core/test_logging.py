import io
import json

import structlog
from pydantic import SecretStr

from docseek.core.logging import REDACTED, configure_logging, get_logger


def _capture(level: str = "INFO") -> io.StringIO:
    stream = io.StringIO()
    configure_logging(level, json=True, stream=stream)
    return stream


def _lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_emits_json_with_level_timestamp_and_logger() -> None:
    stream = _capture()
    get_logger("docseek.test").info("indexed", n_chunks=3)

    (record,) = _lines(stream)
    assert record["event"] == "indexed"
    assert record["n_chunks"] == 3
    assert record["level"] == "info"
    assert record["logger_name"] == "docseek.test"
    assert "timestamp" in record


def test_masks_secret_looking_keys_recursively() -> None:
    stream = _capture()
    get_logger("t").info(
        "call", api_key="sk-1", headers={"Authorization": "Bearer x"}, password=SecretStr("pw")
    )

    text = stream.getvalue()
    for secret in ("sk-1", "Bearer x", "pw"):
        assert secret not in text
    (record,) = _lines(stream)
    assert record["api_key"] == REDACTED
    assert record["headers"] == {"Authorization": REDACTED}


def test_level_filtering() -> None:
    stream = _capture("WARNING")
    log = get_logger("t")
    log.info("hidden")
    log.warning("shown")

    assert [r["event"] for r in _lines(stream)] == ["shown"]


def test_trace_id_from_contextvars() -> None:
    stream = _capture()
    with structlog.contextvars.bound_contextvars(trace_id="tr-1"):
        get_logger("t").info("step")

    assert _lines(stream)[0]["trace_id"] == "tr-1"


def test_logger_created_before_configure_uses_later_config() -> None:
    early = get_logger("early")  # like a module-level logger created at import time
    stream = _capture("WARNING")
    early.info("hidden")
    early.warning("shown")

    assert [(r["event"], r["logger_name"]) for r in _lines(stream)] == [("shown", "early")]
