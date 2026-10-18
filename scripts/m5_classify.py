"""Query-type classifier accuracy against golden ``qtype`` labels.

Scored on queries whose label the classifier can predict from text (permission, scan and
image are excluded: they depend on principal or corpus). Writes
experiments/<ts>_m5_classify/{report.json,predictions.jsonl,traces.jsonl}.

    uv run python scripts/m5_classify.py [--no-llm]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from docseek.core.config import Settings, load_models_config
from docseek.core.runs import environment_info, make_run_dir, write_json
from docseek.core.tracing import JsonlTraceSink, Tracer, read_traces, summarize_llm_costs
from docseek.eval.golden import load_golden
from docseek.llm.client import build_llm_client
from docseek.query.classifier import PREDICTED, QueryClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("data/golden/golden_v1.jsonl"))
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    args = ap.parse_args()
    run = make_run_dir(Path("experiments"), "m5_classify" + ("_rules" if args.no_llm else ""))
    tracer = Tracer(JsonlTraceSink(run / "traces.jsonl"))
    llm = None if args.no_llm else build_llm_client(Settings(), tracer)
    clf = QueryClassifier(llm, load_models_config().llm.query_role)
    rows = []
    for q in load_golden(args.golden):
        if str(q.qtype) not in PREDICTED or (args.split != "all" and q.split != args.split):
            continue
        c = clf.classify(q.text, use_llm=not args.no_llm)
        rows.append(
            {
                "qid": q.qid,
                "split": q.split,
                "gold": str(q.qtype),
                "pred": str(c.qtype),
                "source": c.source,
            }
        )
    with (run / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    def acc(rs: list[dict]) -> float:
        return sum(r["gold"] == r["pred"] for r in rs) / len(rs) if rs else float("nan")

    by_source = {s: [r for r in rows if r["source"] == s] for s in ("rule", "llm", "fallback")}
    report = {
        "n": len(rows),
        "accuracy": acc(rows),
        "accuracy_by_split": {
            s: acc([r for r in rows if r["split"] == s]) for s in ("dev", "test")
        },
        "by_source": {s: {"n": len(rs), "accuracy": acc(rs)} for s, rs in by_source.items()},
        "per_gold_type_recall": {
            t: acc([r for r in rows if r["gold"] == t]) for t in sorted({r["gold"] for r in rows})
        },
        "confusion": Counter(f"{r['gold']}->{r['pred']}" for r in rows if r["gold"] != r["pred"]),
        "cost": summarize_llm_costs(read_traces(run / "traces.jsonl")).__dict__
        if (run / "traces.jsonl").exists()
        else None,
    }
    write_json(run / "report.json", report)
    write_json(
        run / "env.json", environment_info(command="scripts/m5_classify.py", no_llm=args.no_llm)
    )
    print(
        json.dumps(
            {k: report[k] for k in ("n", "accuracy", "accuracy_by_split", "by_source")}, indent=1
        )
    )
    print(f"-> {run}")


if __name__ == "__main__":
    main()
