"""Generate Corpus B from ``configs/internal_kb_plan.yaml`` (``docseek data gen-internal``).

The plan fixes every document's metadata and the facts it must state; the generator LLM
only writes the prose. Each reply is checked (anchored headings, every number and
identifier of every fact present, FastAPI symbols mentioned, sane length) and retried with
feedback when a check fails. Output is committed to ``data/internal_kb/`` and is not
regenerated on later runs unless ``force`` is set.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from docseek.core.config import LLMRoleConfig
from docseek.core.logging import get_logger
from docseek.core.tracing import trace_scope
from docseek.ingest.markdown_parser import (
    ANCHOR_SUFFIX_RE,
    HEADING_RE,
    FenceTracker,
)
from docseek.llm.client import CachedLLMClient, StructuredOutputError

log = get_logger(__name__)

_NUMBER_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d+(?:[.,:]\d+)*%?")
_CODE_RE = re.compile(r"`([^`]+)`")
_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+")
MIN_WORDS, MAX_WORDS = 120, 700


class KBPlanEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    doc_type: str
    acl: list[str]
    owner: str
    version: int = Field(ge=1)
    updated_at: date
    status: Literal["active", "deprecated"] = "active"
    supersedes: str | None = None
    facts: list[str] = Field(min_length=1)
    fastapi_refs: list[str] = Field(default_factory=list)
    table: str | None = None


class KBPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    company: str
    docs: list[KBPlanEntry] = Field(min_length=1)


class KBBody(BaseModel):
    """Structured reply of the generator."""

    markdown: str


def load_plan(path: Path) -> KBPlan:
    with path.open(encoding="utf-8") as fh:
        return KBPlan.model_validate(yaml.safe_load(fh))


def key_terms(fact: str) -> list[str]:
    """Strings a body must contain verbatim for ``fact`` to count as stated.

    Checked: inline-code spans (the plan backticks names such as `ledger-service`),
    identifiers like ``Retry-After``, ``ACME-4001`` or ``starting_after``, and numbers or
    ISO dates outside those. Ordinary words, hyphenated ones like "on-call" included, may
    be paraphrased.
    """
    code = _CODE_RE.findall(fact)
    rest = _CODE_RE.sub(" ", fact)
    idents = [t for t in _TOKEN_RE.findall(rest) if _is_identifier(t)]
    for ident in idents:
        rest = rest.replace(ident, " ")
    out: list[str] = []
    for t in [*code, *idents, *_NUMBER_RE.findall(rest)]:
        t = t.rstrip(".,")
        if t and t not in out:
            out.append(t)
    return out


def _is_identifier(token: str) -> bool:
    if "_" in token:
        return True
    return any(part[:1].isupper() or part[:1].isdigit() for part in token.split("-")[1:])


def check_body(entry: KBPlanEntry, markdown: str) -> list[str]:
    """Problems with a generated body; empty means it is acceptable."""
    problems: list[str] = []
    lines = markdown.strip().splitlines()
    if not lines or not lines[0].startswith("# "):
        problems.append("the first line must be the H1 title")
    if lines and lines[0].strip() == "---":
        problems.append("do not include YAML front matter")
    fences = FenceTracker()
    n_h2 = 0
    for line in lines:
        if fences.feed(line):
            continue
        m = HEADING_RE.match(line)
        if not m:
            continue
        if not ANCHOR_SUFFIX_RE.search(m["body"]):
            problems.append(f"heading without {{ #anchor }}: {line.strip()!r}")
        n_h2 += len(m["hashes"]) == 2
    if n_h2 < 2:
        problems.append("use at least two '## ' sections")
    for fact in entry.facts:
        missing = [t for t in key_terms(fact) if t not in markdown]
        if missing:
            problems.append(f"fact not stated exactly ({', '.join(missing)} missing): {fact}")
    for ref in entry.fastapi_refs:
        if ref not in markdown:
            problems.append(f"mention the FastAPI concept `{ref}`")
    if entry.table and not re.search(r"^\|.*\|\s*$", markdown, re.M):
        problems.append("include the requested markdown table")
    n_words = len(markdown.split())
    if not MIN_WORDS <= n_words <= MAX_WORDS:
        problems.append(f"length is {n_words} words; write {MIN_WORDS}-{MAX_WORDS}")
    return problems


SYSTEM_PROMPT = """\
You write internal engineering documentation for {company}, a payments company whose \
services are built with Python and FastAPI. Write like a careful senior engineer: concrete, \
specific, no marketing language. Never say the company or the document is fictional.

Output rules:
- Markdown only, no YAML front matter.
- The first line is the H1 title followed by an anchor, e.g. `# Title {{ #title }}`.
- Use 2-4 `## ` sections (optionally `### `). EVERY heading ends with `{{ #lowercase-slug }}`.
- 180-450 words in total.
- State every required fact, keeping its numbers, names and identifiers exactly as given.
- Do not add other numbers that contradict the required facts.
"""


def build_messages(
    entry: KBPlanEntry, company: str, problems: list[str] | None = None
) -> list[dict[str, Any]]:
    facts = "\n".join(f"- {f}" for f in entry.facts)
    parts = [
        f"Document type: {entry.doc_type}",
        f"Title: {entry.title}",
        f"Owning team: {entry.owner}",
        f"Audience: {', '.join(entry.acl)}",
        f"Status: {entry.status}"
        + (" (an older version, since replaced)" if entry.status == "deprecated" else ""),
        f"Required facts:\n{facts}",
    ]
    if entry.fastapi_refs:
        refs = ", ".join(f"`{r}`" for r in entry.fastapi_refs)
        parts.append(f"Mention these FastAPI concepts naturally, in backticks: {refs}.")
    if entry.table:
        parts.append(entry.table)
    if entry.doc_type == "postmortem":
        parts.append("Use sections such as Summary, Timeline, Root cause and Action items.")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(company=company)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    if problems:
        fix = "\n".join(f"- {p}" for p in problems)
        messages.append(
            {
                "role": "user",
                "content": f"Your previous draft had these problems:\n{fix}\nWrite it again.",
            }
        )
    return messages


def render_document(entry: KBPlanEntry, body: str) -> str:
    """Front matter (§3.2 fields + doc_type) followed by the generated body."""
    front = {
        "id": entry.id,
        "title": entry.title,
        "doc_type": entry.doc_type,
        "acl": list(entry.acl),
        "version": entry.version,
        "updated_at": entry.updated_at,
        "supersedes": entry.supersedes,
        "status": entry.status,
        "owner": entry.owner,
    }
    dumped = yaml.safe_dump(front, sort_keys=False, default_flow_style=None, allow_unicode=True)
    return f"---\n{dumped}---\n\n{body.strip()}\n"


@dataclass
class GenResult:
    id: str
    status: Literal["written", "skipped", "failed"]
    attempts: int = 0
    problems: list[str] = field(default_factory=list)


def generate_document(
    entry: KBPlanEntry,
    *,
    company: str,
    llm: CachedLLMClient,
    role: LLMRoleConfig,
    out_dir: Path,
    max_attempts: int = 3,
) -> GenResult:
    problems: list[str] = []
    for attempt in range(1, max_attempts + 1):
        messages = build_messages(entry, company, problems or None)
        try:
            resp = llm.complete(
                messages,
                model=role.model,
                temperature=role.temperature,
                response_format=KBBody,
                name="gen_internal.document",
            )
        except StructuredOutputError as exc:
            problems = [f"reply was not valid JSON for the schema: {exc}"]
            continue
        assert resp.parsed is not None
        body = str(resp.parsed["markdown"])
        problems = check_body(entry, body)
        if not problems:
            (out_dir / f"{entry.id}.md").write_text(
                render_document(entry, body), encoding="utf-8", newline="\n"
            )
            return GenResult(entry.id, "written", attempt)
        log.info("gen_internal.retry", id=entry.id, attempt=attempt, problems=problems)
    return GenResult(entry.id, "failed", max_attempts, problems)


def generate_internal_kb(
    plan: KBPlan,
    *,
    llm: CachedLLMClient,
    role: LLMRoleConfig,
    out_dir: Path,
    force: bool = False,
    workers: int = 4,
    max_attempts: int = 3,
) -> list[GenResult]:
    """Write ``<out_dir>/<id>.md`` per planned doc; existing files are kept unless ``force``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [e for e in plan.docs if force or not (out_dir / f"{e.id}.md").exists()]
    skipped = [GenResult(e.id, "skipped") for e in plan.docs if e not in todo]

    def run(entry: KBPlanEntry) -> GenResult:
        with trace_scope():
            return generate_document(
                entry,
                company=plan.company,
                llm=llm,
                role=role,
                out_dir=out_dir,
                max_attempts=max_attempts,
            )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(run, todo))
    order = {e.id: i for i, e in enumerate(plan.docs)}
    return sorted([*skipped, *results], key=lambda r: order[r.id])
