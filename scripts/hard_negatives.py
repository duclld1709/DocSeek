"""Share of (positive, hard negative) pairs a run orders correctly.

For every golden query with hard negatives and every pair (p, h) of a grade>=1 section and a
hard negative, the pair is correct when p is ranked above h (a section missing from the run's
top-k counts as ranked below everything retrieved; both missing = not counted).

    uv run python scripts/hard_negatives.py RUN_DIR... [--golden data/golden/golden_v1.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docseek.eval.golden import load_golden
from docseek.eval.runner import load_per_query


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--golden", type=Path, default=Path("data/golden/golden_v1.jsonl"))
    args = ap.parse_args()
    golden = {q.qid: q for q in load_golden(args.golden) if q.hard_negatives and q.relevant}
    for run in args.runs:
        rows = load_per_query(run)
        correct = total = 0
        for qid, q in golden.items():
            if qid not in rows:
                continue
            pos = {sid: i for i, sid in enumerate(rows[qid]["ranked"])}
            for p in q.relevant:
                for h in q.hard_negatives:
                    rp, rh = pos.get(p), pos.get(h)
                    if rp is None and rh is None:
                        continue
                    total += 1
                    correct += rh is None or (rp is not None and rp < rh)
        name = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["name"]
        share = correct / total if total else float("nan")
        print(f"{name}: {correct}/{total} pairs ordered correctly ({share:.3f})")


if __name__ == "__main__":
    main()
