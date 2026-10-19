"""Answer-mode latency from the first *uncached* execution of each (question, mode).

The final answer run replays earlier calls from the LLM cache (after quota/network stops), so
its wall times are not latencies. This script scans every experiments/*_m8_answers run in
time order, takes the first execution of each (qid, mode) whose LLM calls were all real
(``cached: false``), and reports per mode:

- ``wall_ms``: end-to-end, including client-side waits for the free tier's rate limit;
- ``compute_ms``: LLM call time + tool time from the trace (no rate-limit waits).

    uv run python scripts/m8_answer_latency.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from docseek.core.runs import make_run_dir, write_json
from docseek.core.tracing import read_traces


def pct(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "n": len(values)}


def main() -> None:
    runs = sorted(Path("experiments").glob("*_m8_answers"))
    first: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        events = read_traces(run / "traces.jsonl") if (run / "traces.jsonl").exists() else []
        by_trace: defaultdict[str, list[Any]] = defaultdict(list)
        for e in events:
            by_trace[e.trace_id].append(e)
        answers = run / "answers.jsonl"
        if not answers.exists():
            continue
        for row in map(json.loads, answers.open(encoding="utf-8")):
            key = (row["qid"], row["mode"])
            if key in first:
                continue
            evs = by_trace.get(row["trace_id"], [])
            llm = [e for e in evs if e.kind == "llm_call"]
            if not llm or any(e.output.get("cached") for e in llm):
                continue
            tools = [e for e in evs if e.kind == "tool_call"]
            first[key] = {
                "run": run.name,
                "wall_ms": row["wall_ms"],
                "compute_ms": sum(e.latency_ms for e in llm) + sum(e.latency_ms for e in tools),
                "llm_ms": sum(e.latency_ms for e in llm),
                "tool_ms": sum(e.latency_ms for e in tools),
                "llm_calls": len(llm),
            }
    report: dict[str, Any] = {"runs": [r.name for r in runs], "modes": {}}
    for mode in ("one-shot", "agentic", "agentic-no-fs"):
        rows = [v for (_, m), v in first.items() if m == mode]
        report["modes"][mode] = {
            "n_uncached": len(rows),
            "wall_ms": pct([r["wall_ms"] for r in rows]),
            "compute_ms": pct([r["compute_ms"] for r in rows]),
            "llm_ms": pct([r["llm_ms"] for r in rows]),
            "tool_ms": pct([r["tool_ms"] for r in rows]),
            "mean_llm_calls": float(np.mean([r["llm_calls"] for r in rows])) if rows else None,
        }
    out = make_run_dir(Path("experiments"), "m8_answer_latency")
    write_json(out / "report.json", report)
    print(json.dumps(report["modes"], indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
