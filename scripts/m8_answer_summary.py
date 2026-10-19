"""Summary of an M8 answer run with the right "should abstain" definition.

A question should be abstained on when it is ``unanswerable`` or has no relevant section the
principal may read (permission queries for low-privilege principals). The stage-1 report
counted the latter as answerable; this summary does not.

    uv run python scripts/m8_answer_summary.py experiments/<ts>_m8_answers
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

from docseek.core.runs import make_run_dir, write_json


def main() -> None:
    run = Path(sys.argv[1])
    rows = [json.loads(line) for line in (run / "answers.jsonl").open(encoding="utf-8")]
    out: dict[str, Any] = {"run": run.name, "modes": {}}
    for mode in ("one-shot", "agentic", "agentic-no-fs"):
        rs = [r for r in rows if r["mode"] == mode]
        should = [
            r
            for r in rs
            if r["qtype"] == "unanswerable" or not any(g > 0 for g in r["relevant"].values())
        ]
        ans = [r for r in rs if r not in should]

        def recall(r: dict[str, Any]) -> float:
            rel = {s for s, g in r["relevant"].items() if g > 0}
            return len(rel & set(r["retrieved"][:10])) / len(rel)

        out["modes"][mode] = {
            "n": len(rs),
            "should_abstain": len(should),
            "correct_abstentions": sum(r["abstained"] for r in should),
            "answerable": len(ans),
            "wrong_abstentions": sum(r["abstained"] for r in ans),
            "wrongly_abstained": [r["qid"] for r in ans if r["abstained"]],
            "cites_relevant_rate": sum(bool(r["cited_relevant"]) for r in ans) / len(ans),
            "evidence_recall@10": statistics.mean(recall(r) for r in ans),
            "evidence_hit@10": statistics.mean(
                bool({s for s, g in r["relevant"].items() if g > 0} & set(r["retrieved"][:10]))
                for r in ans
            ),
            "mean_evidence_sections": statistics.mean(len(r["retrieved"]) for r in ans),
            "removed_citations": sum(len(r["removed_citations"]) for r in rs),
            "mean_steps": statistics.mean(r["steps"] for r in rs),
            "max_steps": max(r["steps"] for r in rs),
        }
    dest = make_run_dir(Path("experiments"), "m8_answer_summary")
    write_json(dest / "report.json", out)
    print(json.dumps(out["modes"], indent=1))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
