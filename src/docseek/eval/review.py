"""Human-in-the-loop steps of the golden set (PROJECT_SPEC §5.2 step 5, §5.4).

Files under ``data/golden/`` (all committed, so every number can be traced back):

- ``golden_v1.draft.jsonl``: synthetic draft from ``eval synth``.
- ``golden_v1.review.jsonl``: one decision per reviewed qid (accept / edit / drop).
- ``human_queries_v1.yaml``: queries written by hand (``origin: human``).
- ``golden_v1.jsonl``: draft + review decisions + human queries, validated and split.
- ``calibration_pairs_v1.jsonl``: the fixed sample of (query, section) pairs for calibration.
- ``human_labels_v1.jsonl``: the human's 0/1/2 label per calibration pair.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from docseek.core.acl import AccessPolicy
from docseek.core.models import QueryType, Section
from docseek.eval.golden import GoldenQuery, assign_splits, validate_against_corpus
from docseek.eval.judge import Judged, position_bias, spearman
from docseek.eval.stats import cohen_kappa

# --- review decisions ---------------------------------------------------------------------


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qid: str
    action: Literal["accept", "edit", "drop"]
    text: str | None = None  # new text for "edit"
    relevant: dict[str, int] | None = None  # replaced relevance for "edit"
    note: str | None = None


def read_jsonl_models[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [model.model_validate_json(line) for line in fh if line.strip()]


def append_jsonl(path: Path, row: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(row.model_dump_json() + "\n")


def latest_decisions(decisions: Iterable[ReviewDecision]) -> dict[str, ReviewDecision]:
    """Last decision per qid wins (re-reviewing a query simply appends a new line)."""
    return {d.qid: d for d in decisions}


def review_order(queries: Sequence[GoldenQuery], seed: int = 13) -> list[GoldenQuery]:
    """Round-robin over qtypes (shuffled within each) so any prefix covers every qtype."""
    rng = random.Random(seed)
    by_type: dict[str, list[GoldenQuery]] = {}
    for q in sorted(queries, key=lambda q: q.qid):
        by_type.setdefault(str(q.qtype), []).append(q)
    for group in by_type.values():
        rng.shuffle(group)
    out: list[GoldenQuery] = []
    while any(by_type.values()):
        for t in sorted(by_type):
            if by_type[t]:
                out.append(by_type[t].pop(0))
    return out


def apply_decisions(
    draft: Sequence[GoldenQuery], decisions: dict[str, ReviewDecision]
) -> list[GoldenQuery]:
    out: list[GoldenQuery] = []
    for q in draft:
        d = decisions.get(q.qid)
        if d is None:
            out.append(q)
        elif d.action == "accept":
            out.append(q.model_copy(update={"reviewed": True}))
        elif d.action == "edit":
            update: dict[str, Any] = {"reviewed": True}
            if d.text:
                update["text"] = d.text
                update["lexical_overlap"] = None  # stale after an edit; recomputed on build
            if d.relevant is not None:
                update["relevant"] = d.relevant
            if d.note:
                update["notes"] = d.note
            out.append(GoldenQuery.model_validate({**q.model_dump(), **update}))
        # "drop": left out
    return out


# --- human-written queries ----------------------------------------------------------------


class HumanQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    qtype: QueryType
    principal: str = "anonymous"
    relevant: dict[str, int] = Field(default_factory=dict)
    reference_answer: str | None = None
    stale: list[str] = Field(default_factory=list)
    notes: str | None = None


def load_human_queries(path: Path) -> list[HumanQuery]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return [HumanQuery.model_validate(q) for q in data.get("queries", [])]


def build_golden(
    draft: Sequence[GoldenQuery],
    decisions: dict[str, ReviewDecision],
    human: Sequence[HumanQuery],
    *,
    sections: Sequence[Section],
    policy: AccessPolicy,
    seed: int = 13,
) -> list[GoldenQuery]:
    """Draft + decisions + human queries, validated, renumbered, re-split (stratified)."""
    from docseek.eval.synth import lexical_overlap

    by_id = {s.section_id: s for s in sections}
    merged = apply_decisions(draft, decisions)
    merged += [
        GoldenQuery(
            qid="q9999",  # placeholder, renumbered below
            text=h.text,
            qtype=h.qtype,
            principal=h.principal,
            relevant=h.relevant,
            reference_answer=h.reference_answer,
            origin="human",
            reviewed=True,
            stale=h.stale,
            notes=h.notes,
        )
        for h in human
    ]
    renumbered: list[GoldenQuery] = []
    for i, q in enumerate(merged, start=1):
        overlap = q.lexical_overlap
        if overlap is None and q.relevant and all(s in by_id for s in q.relevant):
            overlap = round(
                lexical_overlap(q.text, "\n".join(by_id[s].text for s in q.relevant)), 4
            )
        renumbered.append(q.model_copy(update={"qid": f"q{i:04d}", "lexical_overlap": overlap}))
    validate_against_corpus(renumbered, {s.section_id: s.acl for s in sections}, policy)
    return assign_splits(renumbered, seed=seed)


def find_sections(sections: Sequence[Section], needle: str, limit: int = 20) -> list[Section]:
    """Sections whose title, path or text contain ``needle`` (case-insensitive); titles first."""
    n = needle.lower()
    in_title = [
        s for s in sections if n in s.title.lower() or n in " ".join(s.section_path).lower()
    ]
    in_text = [s for s in sections if n in s.text.lower() and s not in in_title]
    return [*in_title, *in_text][:limit]


# --- calibration --------------------------------------------------------------------------


class CalibrationPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: str
    qid: str
    query: str
    section_id: str
    stratum: Literal["relevant", "hard_negative", "same_doc_type"]


class HumanLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: str
    score: int = Field(ge=0, le=2)
    note: str | None = None


def sample_calibration_pairs(
    golden: Sequence[GoldenQuery],
    sections: Sequence[Section],
    policy: AccessPolicy,
    *,
    n: int = 100,
    seed: int = 13,
) -> list[CalibrationPair]:
    """A fixed, seeded mix: ~35% golden-relevant, ~35% hard negatives, ~30% other readable
    sections of the same doc_type, so all three grades are well represented."""
    rng = random.Random(seed)
    by_id = {s.section_id: s for s in sections}
    queries = [q for q in sorted(golden, key=lambda q: q.qid) if q.relevant]
    reviewed_first = sorted(queries, key=lambda q: (not q.reviewed, rng.random()))
    quotas = {"relevant": round(n * 0.35), "hard_negative": round(n * 0.35)}
    quotas["same_doc_type"] = n - sum(quotas.values())
    pairs: list[CalibrationPair] = []
    used: set[tuple[str, str]] = set()

    def add(q: GoldenQuery, sid: str, stratum: str) -> None:
        if (q.qid, sid) in used or quotas[stratum] == 0:
            return
        used.add((q.qid, sid))
        quotas[stratum] -= 1
        pairs.append(
            CalibrationPair(
                pair_id=f"p{len(pairs) + 1:03d}",
                qid=q.qid,
                query=q.text,
                section_id=sid,
                stratum=stratum,  # type: ignore[arg-type]
            )
        )

    for q in reviewed_first:
        rel = sorted(q.relevant)
        add(q, rng.choice(rel), "relevant")
        if q.hard_negatives:
            add(q, rng.choice(sorted(q.hard_negatives)), "hard_negative")
        source = by_id[rel[0]]
        same_type = [
            s.section_id
            for s in sections
            if s.doc_type == source.doc_type
            and s.section_id not in q.relevant
            and s.section_id not in q.hard_negatives
            and s.text.strip()
            and policy.can_read(q.principal, s.acl)
        ]
        if same_type:
            add(q, rng.choice(same_type), "same_doc_type")
        if not any(quotas.values()):
            break
    rng.shuffle(pairs)  # the labeller must not see strata in order
    return [p.model_copy(update={"pair_id": f"p{i:03d}"}) for i, p in enumerate(pairs, 1)]


def calibration_report(
    pairs: Sequence[CalibrationPair],
    human: dict[str, HumanLabel],
    judged_qf: dict[str, Judged],
    judged_sf: dict[str, Judged],
    sections: dict[str, Section],
) -> dict[str, Any]:
    """Judge vs human agreement (weighted kappa), confusion matrix and bias checks."""
    ids = [p.pair_id for p in pairs if p.pair_id in human and p.pair_id in judged_qf]
    h = [human[i].score for i in ids]
    j = [judged_qf[i].score for i in ids]
    confusion = np.zeros((3, 3), dtype=int)
    for a, b in zip(h, j, strict=True):
        confusion[a, b] += 1
    by_pair = {p.pair_id: p for p in pairs}
    lengths = [len(sections[by_pair[i].section_id].text.split()) for i in ids]
    report: dict[str, Any] = {
        "n_pairs": len(ids),
        "kappa_linear": cohen_kappa(h, j, weights="linear"),
        "kappa_quadratic": cohen_kappa(h, j, weights="quadratic"),
        "kappa_unweighted": cohen_kappa(h, j, weights="none"),
        "exact_agreement": float(np.mean(np.asarray(h) == np.asarray(j))),
        "confusion_human_rows_judge_cols": confusion.tolist(),
        "human_label_counts": dict(sorted(Counter(h).items())),
        "judge_label_counts": dict(sorted(Counter(j).items())),
        "length_bias_spearman": {
            "judge": spearman(lengths, j),
            "human": spearman(lengths, h),
        },
        "by_stratum": {},
        "rubric_version": next(iter(judged_qf.values())).rubric_version if judged_qf else None,
        "judge_model": next(iter(judged_qf.values())).model if judged_qf else None,
    }
    for stratum in ("relevant", "hard_negative", "same_doc_type"):
        sids = [i for i in ids if by_pair[i].stratum == stratum]
        if len(sids) >= 2:
            report["by_stratum"][stratum] = {
                "n": len(sids),
                "human_mean": float(np.mean([human[i].score for i in sids])),
                "judge_mean": float(np.mean([judged_qf[i].score for i in sids])),
                "exact_agreement": float(
                    np.mean([human[i].score == judged_qf[i].score for i in sids])
                ),
            }
    both = [i for i in ids if i in judged_sf]
    if both:
        pb = position_bias([judged_qf[i].score for i in both], [judged_sf[i].score for i in both])
        report["position_bias"] = pb.__dict__
    disagreements = [
        {
            "pair_id": i,
            "query": by_pair[i].query,
            "section_id": by_pair[i].section_id,
            "human": human[i].score,
            "judge": judged_qf[i].score,
            "judge_reason": judged_qf[i].reason,
            "human_note": human[i].note,
        }
        for i in ids
        if human[i].score != judged_qf[i].score
    ]
    report["disagreements"] = disagreements
    return report


def dump_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")
