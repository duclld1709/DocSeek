"""Markdown tables from run directories, for docs/results/*.md.

    uv run python scripts/results_table.py RUN_DIR... [--ref RUN_DIR] [--metric ndcg@10]

Prints (1) overall metrics with 95% CIs, latency and the paired p-value vs ``--ref``, and
(2) ``--metric`` per qtype. Every number comes from the run directories given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docseek.eval.runner import compare_runs

OVERALL = ("ndcg@10", "recall@10", "mrr@10", "hit@1")


def _m(run: Path) -> dict:
    return json.loads((run / "metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--metric", default="ndcg@10")
    ap.add_argument("--label", action="append", default=[], help="name=RUN_DIR override")
    args = ap.parse_args()
    labels = dict(lbl.split("=", 1)[::-1] for lbl in args.label)

    def name(run: Path) -> str:
        return labels.get(str(run), _m(run)["name"])

    head = ["config", *OVERALL, "leak", "p50 ms"] + (["p vs ref"] if args.ref else [])
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for run in args.runs:
        m = _m(run)
        o = m["overall"]
        cells = [name(run)]
        for k in OVERALL:
            v = o.get(k)
            cells.append(f"{v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]" if v else "-")
        leak = next((v["total"] for k, v in o.items() if k.startswith("leakage@")), "-")
        cells += [str(leak), f"{m['latency_ms']['p50']:.1f}"]
        if args.ref:
            if run.resolve() == args.ref.resolve():
                cells.append("ref")
            else:
                rows, _ = compare_runs(args.ref, run, metrics=[args.metric])
                cells.append(f"{rows[0].p_value:.3f} ({rows[0].delta:+.3f})" if rows else "-")
        print("| " + " | ".join(cells) + " |")
    print()
    qtypes = sorted({t for run in args.runs for t in _m(run)["by_qtype"]})
    qtypes = [t for t in qtypes if args.metric in _m(args.runs[0])["by_qtype"].get(t, {})]
    first = _m(args.runs[0])["by_qtype"]
    print(
        f"| {args.metric} by qtype | "
        + " | ".join(f"{t} (n={first[t]['n_queries']})" for t in qtypes)
        + " |"
    )
    print("|" + "---|" * (len(qtypes) + 1))
    for run in args.runs:
        bq = _m(run)["by_qtype"]
        cells = [
            f"{bq[t][args.metric]['mean']:.3f}" if args.metric in bq.get(t, {}) else "-"
            for t in qtypes
        ]
        print(f"| {name(run)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
