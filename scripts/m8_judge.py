"""M8 answer eval, stage 2: DeepEval metrics on stage-1 answers (judge role, cached).

    uv run python scripts/m8_judge.py experiments/<ts>_m8_answers [--limit N] [--budget 0.70]

Scores only answerable questions that were answered (abstentions are counted, not judged).
Writes <run>/judged.jsonl, judge_report.json and judge_traces.jsonl (cost from traces).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from docseek.core.config import Settings, load_models_config
from docseek.core.runs import write_json
from docseek.core.tracing import JsonlTraceSink, Tracer, read_traces, trace_scope
from docseek.eval.answer_judge import DocSeekJudge, build_metrics, score_answer
from docseek.eval.stats import bootstrap_ci
from docseek.llm.client import build_llm_client


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget", type=float, default=0.70, help="USD cap for this run")
    args = ap.parse_args()
    rows = [json.loads(line) for line in (args.run / "answers.jsonl").open(encoding="utf-8")]
    todo = [r for r in rows if r["qtype"] != "unanswerable" and not r["abstained"]]
    todo = todo[: args.limit]
    tracer = Tracer(JsonlTraceSink(args.run / "judge_traces.jsonl"))
    settings = Settings().model_copy(update={"budget_usd": args.budget})
    llm = build_llm_client(settings, tracer)
    judge_model = load_models_config().llm.judge.model
    metrics = build_metrics(DocSeekJudge(llm, judge_model))
    out_path = args.run / "judged.jsonl"
    done = (
        {(j["qid"], j["mode"]) for j in map(json.loads, out_path.open(encoding="utf-8"))}
        if out_path.exists()
        else set()
    )
    start = time.perf_counter()
    with out_path.open("a", encoding="utf-8", newline="\n") as fh:
        for i, r in enumerate(todo, start=1):
            if (r["qid"], r["mode"]) in done:
                continue
            with trace_scope():
                scores = score_answer(
                    metrics,
                    question=r["question"],
                    answer=r["answer"],
                    contexts=r["contexts"],
                    reference=r["reference_answer"],
                )
            fh.write(json.dumps({"qid": r["qid"], "mode": r["mode"], **scores}) + "\n")
            fh.flush()
            eta = (time.perf_counter() - start) / i * (len(todo) - i)
            print(
                f"[{i}/{len(todo)}] {r['mode']} {r['qid']} {scores} eta={eta / 60:.0f}min",
                flush=True,
            )
    judged = [json.loads(line) for line in out_path.open(encoding="utf-8")]
    events = [e for e in read_traces(args.run / "judge_traces.jsonl") if e.kind == "llm_call"]
    report: dict[str, Any] = {
        "judge_model": judge_model,
        "judge_calls": len(events),
        "judge_cost_usd": sum(e.cost_usd for e in events),
        "n_judged": len(judged),
        "modes": {},
    }
    by_mode: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for j in judged:
        by_mode[j["mode"]].append(j)
    for mode, js in by_mode.items():
        entry: dict[str, Any] = {"n": len(js)}
        for m in ("faithfulness", "answer_relevancy", "contextual_recall", "citation_support"):
            vals = [j[m] for j in js if j.get(m) is not None]
            if vals:
                ci = bootstrap_ci(vals, seed=13)
                entry[m] = {"mean": statistics.mean(vals), "ci": [ci.low, ci.high], "n": len(vals)}
        report["modes"][mode] = entry
    write_json(args.run / "judge_report.json", report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
