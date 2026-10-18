"""Golden set schema, validation, splits and I/O (PROJECT_SPEC §5.1).

``data/golden/golden_v{N}.jsonl`` holds one ``GoldenQuery`` per line. Relevance is graded per
**section** (2 = answers directly, 1 = related/partial); chunk hits are mapped back to sections
at evaluation time (ADR 0002). Fields beyond the spec's example (``style``, ``source_section``,
``hard_negatives``, ``stale``, ``lexical_overlap``, ``notes``) carry provenance for analysis.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from docseek.core.acl import AccessPolicy
from docseek.core.models import Query, QueryType

Split = Literal["dev", "test"]
QID_RE = re.compile(r"^q\d{4,}$")
DEV_FRACTION = 0.4
SPLIT_SEED = 13
# A permission query asked by a principal without access has nothing it may be shown.
EMPTY_OK = frozenset({QueryType.UNANSWERABLE, QueryType.PERMISSION})


class GoldenQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qid: str
    text: str = Field(min_length=1)
    qtype: QueryType
    principal: str = "anonymous"
    split: Split = "dev"
    relevant: dict[str, int] = Field(default_factory=dict)
    reference_answer: str | None = None
    # synthetic: LLM pipeline; human: written by a person; claude: written by the coding
    # assistant (Claude) on the owner's request. ``annotator`` says who reviewed or wrote it.
    origin: Literal["synthetic", "human", "claude"]
    reviewed: bool = False
    annotator: str | None = None
    style: Literal["human", "machine", "paraphrase"] | None = None
    source_section: str | None = None
    hard_negatives: list[str] = Field(default_factory=list)
    stale: list[str] = Field(default_factory=list)  # superseded sections (freshness queries)
    lexical_overlap: float | None = Field(default=None, ge=0, le=1)
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not QID_RE.match(self.qid):
            raise ValueError(f"qid must look like q0042, got {self.qid!r}")
        if any(g not in (1, 2) for g in self.relevant.values()):
            raise ValueError(f"{self.qid}: relevance grades must be 1 or 2")
        if self.qtype == QueryType.UNANSWERABLE and self.relevant:
            raise ValueError(f"{self.qid}: unanswerable queries have no relevant sections")
        if not self.relevant and self.qtype not in EMPTY_OK:
            raise ValueError(
                f"{self.qid}: only unanswerable or permission queries may have no relevant sections"
            )
        if overlap := set(self.relevant) & set(self.hard_negatives):
            raise ValueError(f"{self.qid}: sections both relevant and hard negative: {overlap}")
        return self

    def to_query(self) -> Query:
        return Query(qid=self.qid, text=self.text, principal=self.principal, qtype=self.qtype)


class GoldenValidationError(ValueError):
    pass


def validate_against_corpus(
    queries: Iterable[GoldenQuery],
    section_acl: Mapping[str, Iterable[str]],
    policy: AccessPolicy,
) -> None:
    """Every referenced section exists and every relevant one is readable by the principal.

    A relevant section the principal cannot read would reward leaking, so it is an error.
    """
    seen: set[str] = set()
    for q in queries:
        if q.qid in seen:
            raise GoldenValidationError(f"duplicate qid {q.qid}")
        seen.add(q.qid)
        refs = [*q.relevant, *q.hard_negatives, *q.stale]
        if q.source_section:
            refs.append(q.source_section)
        if missing := [s for s in refs if s not in section_acl]:
            raise GoldenValidationError(f"{q.qid}: unknown sections {missing}")
        hidden = [s for s in q.relevant if not policy.can_read(q.principal, section_acl[s])]
        if hidden:
            raise GoldenValidationError(f"{q.qid}: {q.principal} cannot read relevant {hidden}")


def assign_splits(
    queries: list[GoldenQuery], *, dev_fraction: float = DEV_FRACTION, seed: int = SPLIT_SEED
) -> list[GoldenQuery]:
    """Stratified dev/test split per ``qtype`` (dev ≈ 40%), deterministic for a seed."""
    by_type: defaultdict[QueryType, list[GoldenQuery]] = defaultdict(list)
    for q in sorted(queries, key=lambda q: q.qid):
        by_type[q.qtype].append(q)
    split_of: dict[str, Split] = {}
    rng = random.Random(seed)
    for qtype in sorted(by_type):
        group = by_type[qtype]
        rng.shuffle(group)
        n_dev = round(len(group) * dev_fraction)
        for i, q in enumerate(group):
            split_of[q.qid] = "dev" if i < n_dev else "test"
    return [q.model_copy(update={"split": split_of[q.qid]}) for q in queries]


def load_golden(path: Path) -> list[GoldenQuery]:
    with path.open(encoding="utf-8") as fh:
        return [GoldenQuery.model_validate_json(line) for line in fh if line.strip()]


def write_golden(path: Path, queries: Iterable[GoldenQuery]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(queries, key=lambda q: q.qid)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for q in rows:
            fh.write(q.model_dump_json(exclude_defaults=False) + "\n")


def select(
    queries: Iterable[GoldenQuery],
    *,
    split: Split | Literal["all"] = "all",
    reviewed_only: bool = False,
) -> list[GoldenQuery]:
    return [
        q
        for q in queries
        if (split == "all" or q.split == split) and (q.reviewed or not reviewed_only)
    ]
