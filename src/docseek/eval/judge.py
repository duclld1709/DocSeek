"""LLM-as-judge for (query, section) relevance on the 0/1/2 scale (PROJECT_SPEC §5.4).

The rubric lives in ``eval/prompts/relevance_judge.md``; its sha256 is recorded with every
judgment so labels from different rubric revisions are never mixed silently. Section text is
wrapped in delimiters and the rubric tells the model to treat it as data (documents could
contain prompt-injection text).

Bias checks:

- position: the same pair judged with the query before vs after the section;
- length: Spearman correlation between section length and score;
- self-preference: avoided by config (the judge's provider differs from the generator's).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from docseek.core.config import LLMRoleConfig
from docseek.core.models import Section
from docseek.eval.stats import cohen_kappa
from docseek.llm.client import CachedLLMClient

Order = Literal["query_first", "section_first"]
MAX_SECTION_CHARS = 6000


class RelevanceJudgment(BaseModel):
    score: int = Field(ge=0, le=2)
    reason: str


def load_rubric() -> str:
    return (resources.files("docseek.eval") / "prompts" / "relevance_judge.md").read_text(
        encoding="utf-8"
    )


def rubric_version(rubric: str) -> str:
    return hashlib.sha256(rubric.encode("utf-8")).hexdigest()[:12]


def render_section(section: Section, max_chars: int = MAX_SECTION_CHARS) -> str:
    text = section.text if len(section.text) <= max_chars else section.text[:max_chars] + " [...]"
    status = f' status="{section.status}"' if section.status != "active" else ""
    path = " > ".join(section.section_path)
    return f'<section title="{section.title}" path="{path}"{status}>\n{text}\n</section>'


def build_judge_messages(
    query: str, section: Section, rubric: str, *, order: Order = "query_first"
) -> list[dict[str, str]]:
    q = f"<query>\n{query}\n</query>"
    s = render_section(section)
    body = f"{q}\n\n{s}" if order == "query_first" else f"{s}\n\n{q}"
    return [
        {"role": "system", "content": rubric},
        {"role": "user", "content": f"{body}\n\nGrade the section for the query."},
    ]


@dataclass(frozen=True)
class Judged:
    score: int
    reason: str
    rubric_version: str
    model: str


class RelevanceJudge:
    """Grades (query, section) pairs with the judge model from ``configs/models.yaml``."""

    def __init__(
        self, llm: CachedLLMClient, role: LLMRoleConfig, rubric: str | None = None
    ) -> None:
        self.llm = llm
        self.role = role
        self.rubric = rubric if rubric is not None else load_rubric()
        self.version = rubric_version(self.rubric)

    def judge(self, query: str, section: Section, *, order: Order = "query_first") -> Judged:
        resp = self.llm.complete(
            build_judge_messages(query, section, self.rubric, order=order),
            model=self.role.model,
            temperature=self.role.temperature,
            response_format=RelevanceJudgment,
            name="judge.relevance",
        )
        assert resp.parsed is not None
        parsed = RelevanceJudgment.model_validate(resp.parsed)
        return Judged(parsed.score, parsed.reason, self.version, self.role.model)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rho with average ranks for ties; 0.0 when either side is constant."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two sequences of the same length >= 2")
    rx, ry = _ranks(np.asarray(x, float)), _ranks(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _ranks(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    order = a.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    for value in np.unique(a):
        mask = a == value
        ranks[mask] = ranks[mask].mean()
    return ranks


@dataclass(frozen=True)
class PositionBias:
    n: int
    agreement: float  # share of pairs with the same score in both orders
    kappa: float  # linear-weighted kappa between the two orders
    mean_shift: float  # mean(score_section_first - score_query_first)


def position_bias(query_first: Sequence[int], section_first: Sequence[int]) -> PositionBias:
    if len(query_first) != len(section_first) or not query_first:
        raise ValueError("need paired, non-empty score lists")
    a, b = np.asarray(query_first), np.asarray(section_first)
    return PositionBias(
        n=len(a),
        agreement=float((a == b).mean()),
        kappa=cohen_kappa(list(a), list(b)),
        mean_shift=float((b - a).mean()),
    )
