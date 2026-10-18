"""Request tracing: every step of a query (or of an offline job) becomes a ``TraceEvent``.

``trace_scope()`` binds a ``trace_id`` to the current context (thread/task), so deeper layers
such as the LLM client attach their events to the right trace without passing ids around.
The id is also bound into structlog's context variables, so log lines carry it too.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import structlog

from docseek.core.models import TraceEvent, TraceKind

_current_trace: ContextVar[str | None] = ContextVar("docseek_trace_id", default=None)


def new_trace_id() -> str:
    return uuid.uuid4().hex


def current_trace_id() -> str | None:
    return _current_trace.get()


@contextmanager
def trace_scope(trace_id: str | None = None) -> Iterator[str]:
    """Bind ``trace_id`` (or a fresh one) for the duration of the block."""
    tid = trace_id or new_trace_id()
    token = _current_trace.set(tid)
    with structlog.contextvars.bound_contextvars(trace_id=tid):
        try:
            yield tid
        finally:
            _current_trace.reset(token)


class TraceSink(Protocol):
    def write(self, event: TraceEvent) -> None: ...


class MemoryTraceSink:
    """Keeps events in memory (tests, and runs that write traces at the end)."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self._lock = threading.Lock()

    def write(self, event: TraceEvent) -> None:
        with self._lock:
            self.events.append(event)


class JsonlTraceSink:
    """Appends one JSON object per event to ``path`` (created with its parents)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: TraceEvent) -> None:
        line = event.model_dump_json()
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")


class Tracer:
    """Creates trace events with per-trace step numbers and hands them to a sink."""

    def __init__(self, sink: TraceSink | None = None) -> None:
        self.sink: TraceSink = sink if sink is not None else MemoryTraceSink()
        self._steps: defaultdict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(
        self,
        *,
        kind: TraceKind,
        name: str,
        input: dict[str, Any],
        output: dict[str, Any],
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        trace_id: str | None = None,
    ) -> TraceEvent:
        tid = trace_id or current_trace_id() or new_trace_id()
        with self._lock:
            step = self._steps[tid]
            self._steps[tid] += 1
        event = TraceEvent(
            trace_id=tid,
            step=step,
            kind=kind,
            name=name,
            input=input,
            output=output,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )
        self.sink.write(event)
        return event


def read_traces(path: Path) -> list[TraceEvent]:
    """Load events written by ``JsonlTraceSink``."""
    with path.open(encoding="utf-8") as fh:
        return [TraceEvent.model_validate(json.loads(line)) for line in fh if line.strip()]


@dataclass(frozen=True)
class CostSummary:
    llm_calls: int
    cached_calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    by_name: dict[str, float]


def summarize_llm_costs(events: list[TraceEvent]) -> CostSummary:
    """Totals over ``llm_call`` events; ``cost_usd`` is money actually spent (cache hits = 0)."""
    calls = [e for e in events if e.kind == "llm_call"]
    by_name: defaultdict[str, float] = defaultdict(float)
    for e in calls:
        by_name[e.name] += e.cost_usd
    return CostSummary(
        llm_calls=len(calls),
        cached_calls=sum(1 for e in calls if e.output.get("cached")),
        tokens_in=sum(e.tokens_in for e in calls),
        tokens_out=sum(e.tokens_out for e in calls),
        cost_usd=sum(e.cost_usd for e in calls),
        by_name=dict(by_name),
    )
