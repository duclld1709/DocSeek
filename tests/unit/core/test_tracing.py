"""Tracing: trace scopes, step numbering, JSONL round trip and cost summaries."""

import threading
from pathlib import Path

from docseek.core.tracing import (
    JsonlTraceSink,
    MemoryTraceSink,
    Tracer,
    current_trace_id,
    read_traces,
    summarize_llm_costs,
    trace_scope,
)


def _record(tracer: Tracer, **kw: object) -> None:
    args: dict = {"kind": "llm_call", "name": "x", "input": {}, "output": {}, "latency_ms": 1.0}
    args.update(kw)
    tracer.record(**args)


def test_trace_scope_binds_and_restores_ids() -> None:
    assert current_trace_id() is None
    with trace_scope("outer") as outer:
        assert outer == current_trace_id() == "outer"
        with trace_scope() as inner:
            assert current_trace_id() == inner != "outer"
        assert current_trace_id() == "outer"
    assert current_trace_id() is None


def test_steps_are_numbered_per_trace() -> None:
    sink = MemoryTraceSink()
    tracer = Tracer(sink)
    with trace_scope("a"):
        _record(tracer)
        _record(tracer)
    _record(tracer, trace_id="b")
    with trace_scope("a"):
        _record(tracer)
    assert [(e.trace_id, e.step) for e in sink.events] == [("a", 0), ("a", 1), ("b", 0), ("a", 2)]


def test_events_without_scope_get_fresh_trace_ids() -> None:
    sink = MemoryTraceSink()
    tracer = Tracer(sink)
    _record(tracer)
    _record(tracer)
    assert sink.events[0].trace_id != sink.events[1].trace_id


def test_jsonl_round_trip_and_thread_safety(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "traces.jsonl"
    tracer = Tracer(JsonlTraceSink(path))

    def work(i: int) -> None:
        with trace_scope(f"t{i}"):
            for _ in range(20):
                _record(tracer, cost_usd=0.001)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    events = read_traces(path)
    assert len(events) == 100
    for i in range(5):
        assert sorted(e.step for e in events if e.trace_id == f"t{i}") == list(range(20))


def test_summarize_llm_costs_counts_only_llm_calls() -> None:
    sink = MemoryTraceSink()
    tracer = Tracer(sink)
    _record(tracer, name="synth", cost_usd=0.5, tokens_in=10, tokens_out=5)
    _record(tracer, name="synth", cost_usd=0.0, output={"cached": True}, tokens_in=10)
    _record(tracer, name="judge", cost_usd=0.25)
    _record(tracer, kind="retrieve", name="bm25", cost_usd=9.0)
    s = summarize_llm_costs(sink.events)
    assert (s.llm_calls, s.cached_calls, s.tokens_in, s.tokens_out) == (3, 1, 20, 5)
    assert s.cost_usd == 0.75
    assert s.by_name == {"synth": 0.5, "judge": 0.25}
