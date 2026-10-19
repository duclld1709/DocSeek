"""M8 answer eval, stage 1: answer ~60 stratified test queries in each answer mode.

    uv run python scripts/m8_answer_eval.py [--modes one-shot agentic agentic-no-fs] [--limit N]

Writes experiments/<ts>_m8_answers/{answers.jsonl, traces.jsonl, report.json, env.json}.
answers.jsonl is appended per answer (a quota stop keeps what was done; reruns hit the cache).
Stage 2 (DeepEval) is scripts/m8_judge.py.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from docseek.agent.runtime import ANSWER_MODES, build_runtime
from docseek.core.config import Settings, load_models_config
from docseek.core.runs import environment_info, make_run_dir, write_json
from docseek.core.tracing import JsonlTraceSink, Tracer, read_traces, trace_scope
from docseek.eval.golden import GoldenQuery, load_golden
from docseek.llm.client import build_llm_client

# Answerable test queries per qtype (48) + 12 unanswerable = 60.
QUOTA = {
    "factoid": 8,
    "howto": 10,
    "conceptual": 6,
    "symbol": 6,
    "multihop": 5,
    "freshness": 4,
    "permission": 3,
    "scan": 3,
    "image": 3,
    "unanswerable": 12,
}
CONTEXT_CHARS = 1500


def select_queries(golden: list[GoldenQuery], seed: int) -> list[GoldenQuery]:
    by_type: defaultdict[str, list[GoldenQuery]] = defaultdict(list)
    for q in sorted(golden, key=lambda q: q.qid):
        if q.split == "test":
            by_type[str(q.qtype)].append(q)
    rng = random.Random(seed)
    out: list[GoldenQuery] = []
    for qtype, n in QUOTA.items():
        pool = by_type[qtype]
        out += rng.sample(pool, min(n, len(pool)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(ANSWER_MODES))
    ap.add_argument("--golden", type=Path, default=Path("data/golden/golden_m7.jsonl"))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=None, help="first N selected queries (pilot)")
    args = ap.parse_args()

    queries = select_queries(load_golden(args.golden), args.seed)[: args.limit]
    run = make_run_dir(Path("experiments"), "m8_answers")
    tracer = Tracer(JsonlTraceSink(run / "traces.jsonl"))
    llm = build_llm_client(Settings(), tracer)
    model = load_models_config().llm.agent.model
    rt = build_runtime()
    print(f"{len(queries)} queries x {len(args.modes)} modes; disabled: {rt.disabled}", flush=True)
    write_json(
        run / "env.json",
        environment_info(
            command="scripts/m8_answer_eval.py",
            model=model,
            modes=args.modes,
            seed=args.seed,
            qids=[q.qid for q in queries],
            disabled_tools=rt.disabled,
        ),
    )
    rows: list[dict[str, Any]] = []
    total, done, started = len(queries) * len(args.modes), 0, time.perf_counter()
    with (run / "answers.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        for mode in args.modes:
            for q in queries:
                with trace_scope() as tid:
                    t = time.perf_counter()
                    res = rt.answer(mode, q.text, q.principal, llm, model)
                    wall = (time.perf_counter() - t) * 1000
                cited = res.citations or res.retrieved[:5]
                contexts = []
                for sid in dict.fromkeys([*res.citations, *res.retrieved[:5]]):
                    chunks = rt.backend.sections.get(sid) or []
                    text = " ".join(" ".join(c.text for c in chunks).split())[:CONTEXT_CHARS]
                    contexts.append(f"{sid}: {text}")
                row = {
                    "qid": q.qid,
                    "qtype": str(q.qtype),
                    "principal": q.principal,
                    "mode": mode,
                    "question": q.text,
                    "reference_answer": q.reference_answer,
                    "relevant": q.relevant,
                    "answer": res.answer,
                    "abstained": res.abstained,
                    "citations": res.citations,
                    "removed_citations": res.removed_citations,
                    "cited_relevant": sorted(set(res.citations) & set(q.relevant)),
                    "retrieved": res.retrieved,
                    "steps": res.steps,
                    "tool_calls": [c["name"] for c in res.tool_calls],
                    "stop_reason": res.stop_reason,
                    "wall_ms": wall,
                    "trace_id": tid,
                    "contexts": contexts,
                    "context_ids": cited,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                rows.append(row)
                done += 1
                eta = (time.perf_counter() - started) / done * (total - done)
                print(
                    f"[{done}/{total}] {mode} {q.qid} {q.qtype} steps={res.steps} "
                    f"abstain={res.abstained} {wall / 1000:.1f}s eta={eta / 60:.0f}min",
                    flush=True,
                )
    events = read_traces(run / "traces.jsonl")
    by_trace: defaultdict[str, list[Any]] = defaultdict(list)
    for e in events:
        by_trace[e.trace_id].append(e)
    report: dict[str, Any] = {"n_queries": len(queries), "model": model, "modes": {}}
    for mode in args.modes:
        rs = [r for r in rows if r["mode"] == mode]
        ans = [r for r in rs if r["qtype"] != "unanswerable"]
        una = [r for r in rs if r["qtype"] == "unanswerable"]
        llm_events = [e for r in rs for e in by_trace[r["trace_id"]] if e.kind == "llm_call"]
        lat = sorted(r["wall_ms"] for r in rs)
        report["modes"][mode] = {
            "n": len(rs),
            "abstain_rate_answerable": sum(r["abstained"] for r in ans) / max(1, len(ans)),
            "abstain_rate_unanswerable": sum(r["abstained"] for r in una) / max(1, len(una)),
            "cites_relevant_rate": sum(bool(r["cited_relevant"]) for r in ans) / max(1, len(ans)),
            "removed_citations": sum(len(r["removed_citations"]) for r in rs),
            "mean_steps": statistics.mean(r["steps"] for r in rs) if rs else None,
            "max_steps_seen": max((r["steps"] for r in rs), default=0),
            "stop_reasons": dict(Counter(r["stop_reason"] for r in rs)),
            "tools": dict(Counter(t for r in rs for t in r["tool_calls"])),
            "llm_calls": len(llm_events),
            "llm_cost_usd_list": sum(
                e.output.get("original_cost_usd", e.cost_usd) for e in llm_events
            ),
            "tokens_in": sum(e.tokens_in for e in llm_events),
            "tokens_out": sum(e.tokens_out for e in llm_events),
            "latency_ms_p50": lat[len(lat) // 2] if lat else None,
            "latency_ms_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
        }
    write_json(run / "report.json", report)
    print(json.dumps(report["modes"], indent=1))
    print(f"-> {run}")


if __name__ == "__main__":
    main()
