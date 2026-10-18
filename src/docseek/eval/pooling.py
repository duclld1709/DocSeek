"""Golden set v2 by TREC-style pooling (PROJECT_SPEC §5.3).

1. Pool the top-``depth`` sections of several runs per query (from their ``per_query.jsonl``).
2. Pairs without a v1 label (not relevant, not a hard negative, not stale) are graded by the
   LLM judge; every judgment is stored with ``reviewed: false``.
3. v2 = v1 labels (never overridden) + judged sections with grade >= 1.

Unanswerable queries are not pooled: a judged hit would contradict their v1 definition.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from docseek.core.models import QueryType
from docseek.eval.golden import GoldenQuery


@dataclass(frozen=True)
class PoolJudgment:
    qid: str
    section_id: str
    score: int
    reason: str
    model: str
    rubric_version: str
    pooled_from: list[str]
    reviewed: bool = False


def pool_sections(run_dirs: Iterable[Path], depth: int) -> dict[str, dict[str, list[str]]]:
    """``qid -> section_id -> [run names that retrieved it in their top-depth]``."""
    pool: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for run in run_dirs:
        name = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["name"]
        with (run / "per_query.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                for sid in row["ranked"][:depth]:
                    pool[row["qid"]].setdefault(sid, []).append(name)
    return pool


def unlabelled_pairs(
    golden: Sequence[GoldenQuery], pool: Mapping[str, Mapping[str, list[str]]]
) -> list[tuple[GoldenQuery, str, list[str]]]:
    out: list[tuple[GoldenQuery, str, list[str]]] = []
    for q in golden:
        if q.qtype == QueryType.UNANSWERABLE:
            continue
        known = set(q.relevant) | set(q.hard_negatives) | set(q.stale)
        for sid, runs in sorted(pool.get(q.qid, {}).items()):
            if sid not in known:
                out.append((q, sid, sorted(set(runs))))
    return out


JudgeFn = Callable[[str, str], tuple[int, str, str, str]]
"""(query text, section_id) -> (score, reason, model, rubric_version)."""


def judge_pairs(
    pairs: Sequence[tuple[GoldenQuery, str, list[str]]],
    judge: JudgeFn,
    *,
    done: Mapping[tuple[str, str], PoolJudgment] | None = None,
    on_result: Callable[[PoolJudgment], None] | None = None,
) -> list[PoolJudgment]:
    out: list[PoolJudgment] = []
    for q, sid, runs in pairs:
        prior = (done or {}).get((q.qid, sid))
        if prior is not None:
            out.append(prior)
            continue
        score, reason, model, version = judge(q.text, sid)
        j = PoolJudgment(q.qid, sid, score, reason, model, version, runs)
        out.append(j)
        if on_result:
            on_result(j)
    return out


def build_v2(golden: Sequence[GoldenQuery], judgments: Iterable[PoolJudgment]) -> list[GoldenQuery]:
    added: dict[str, dict[str, int]] = defaultdict(dict)
    for j in judgments:
        if j.score >= 1:
            added[j.qid][j.section_id] = j.score
    out: list[GoldenQuery] = []
    for q in golden:
        extra = {sid: g for sid, g in added.get(q.qid, {}).items() if sid not in q.relevant}
        if not extra:
            out.append(q)
            continue
        note = f"v2: +{len(extra)} pooled judge labels (unreviewed)"
        out.append(
            q.model_copy(
                update={
                    "relevant": {**q.relevant, **extra},
                    "notes": f"{q.notes}; {note}" if q.notes else note,
                }
            )
        )
    return out


def write_judgments(path: Path, judgments: Iterable[PoolJudgment]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for j in judgments:
            fh.write(json.dumps(asdict(j), ensure_ascii=False) + "\n")


def read_judgments(path: Path) -> dict[tuple[str, str], PoolJudgment]:
    if not path.is_file():
        return {}
    out: dict[tuple[str, str], PoolJudgment] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                j = PoolJudgment(**json.loads(line))
                out[(j.qid, j.section_id)] = j
    return out
