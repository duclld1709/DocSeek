"""Answer post-processing shared by one-shot RAG and the agent.

- Citations are ``[section_id]`` (several may share brackets: ``[a; b]``).
- **Post-check**: a citation survives only if that section was shown to the model in this
  request (``evidence``); others are removed from the text and reported.
- **Abstain**: the model may answer with the abstain sentence, and an answer left without any
  valid citation becomes an abstention too; both come with the closest sections seen.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

ABSTAIN_TEXT = "I couldn't find this in the docs."
_BRACKET = re.compile(r"\[([^\[\]\n]{3,400})\]")
_ID = re.compile(r"^[a-z0-9_-]+:[^\s#\[\]]+#[^\s\[\]]+$")


class AnswerResult(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    removed_citations: list[str] = Field(default_factory=list)
    abstained: bool = False
    suggestions: list[dict[str, str]] = Field(default_factory=list)
    retrieved: list[str] = Field(default_factory=list)  # evidence, first-seen order
    mode: str = ""
    steps: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    trace_id: str | None = None
    stop_reason: str = "answer"  # answer | max_steps | timeout | budget | degraded


def extract_citations(text: str) -> list[str]:
    out: list[str] = []
    for m in _BRACKET.finditer(text):
        for part in re.split(r"[;,]\s*", m.group(1)):
            part = part.strip()
            if _ID.match(part) and part not in out:
                out.append(part)
    return out


def _strip(text: str, bad: set[str]) -> str:
    def fix(m: re.Match[str]) -> str:
        parts = [p.strip() for p in re.split(r"[;,]\s*", m.group(1))]
        ids = [p for p in parts if _ID.match(p)]
        if not ids:
            return m.group(0)  # not a citation bracket (e.g. a code index)
        keep = [p for p in ids if p not in bad]
        return f"[{'; '.join(keep)}]" if keep else ""

    return re.sub(r"[ \t]+([.,;:])", r"\1", _BRACKET.sub(fix, text)).strip()


def is_abstention(text: str) -> bool:
    t = text.strip().lower()
    return not t or t.startswith("i couldn't find") or t.startswith("i could not find")


def finalize(text: str, evidence: Mapping[str, str], **fields: Any) -> AnswerResult:
    """Apply the citation post-check and the abstain rule to a model answer."""
    cited = extract_citations(text)
    removed = [c for c in cited if c not in evidence]
    kept = [c for c in cited if c in evidence]
    suggestions = [{"section_id": s, "title": t} for s, t in list(evidence.items())[:3]]
    retrieved = list(evidence)
    if is_abstention(text) or not kept:
        return AnswerResult(
            answer=ABSTAIN_TEXT,
            removed_citations=removed,
            abstained=True,
            suggestions=suggestions,
            retrieved=retrieved,
            **fields,
        )
    return AnswerResult(
        answer=_strip(text, set(removed)),
        citations=kept,
        removed_citations=removed,
        retrieved=retrieved,
        **fields,
    )
