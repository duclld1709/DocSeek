"""Corpus B: the simulated "Acme Payments" internal KB (PROJECT_SPEC §3.2).

Each document is ``<dir>/<id>.md`` with YAML front matter::

    id: kb-0012
    title: "Rotating JWT signing keys"
    doc_type: runbook             # one of corpus.yaml internal_kb.doc_types
    acl: ["eng-internal"]         # groups from principals.yaml
    version: 3
    updated_at: 2026-08-02
    supersedes: kb-0007           # older doc on the same topic, or null
    status: active                # active | deprecated
    owner: platform-team

``doc_type`` is an addition to the spec's list of fields: KB docs need one like every other
section, and the spec's FastAPI-folder doc types do not apply here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from docseek.core.config import InternalKBConfig, PrincipalsConfig
from docseek.core.ids import make_doc_id
from docseek.core.models import Section
from docseek.ingest.markdown_parser import PageMeta, ParseIssue, front_matter_end, parse_page

KB_ID_RE = re.compile(r"^kb-\d{4}$")


class KBFrontMatter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str = Field(min_length=1)
    doc_type: str
    acl: list[str] = Field(min_length=1)
    version: int = Field(ge=1)
    updated_at: date
    supersedes: str | None = None
    status: Literal["active", "deprecated"] = "active"
    owner: str = Field(min_length=1)

    @field_validator("id", "supersedes")
    @classmethod
    def _check_id(cls, value: str | None) -> str | None:
        if value is not None and not KB_ID_RE.match(value):
            raise ValueError(f"KB ids look like kb-0012, got {value!r}")
        return value

    @model_validator(mode="after")
    def _not_self(self) -> Self:
        if self.supersedes == self.id:
            raise ValueError(f"{self.id} cannot supersede itself")
        return self


class KBValidationError(ValueError):
    """The KB as a whole is inconsistent (ids, ACL groups, supersedes chain)."""


@dataclass
class KBDocument:
    front: KBFrontMatter
    body: str
    path: Path


def split_front_matter(text: str) -> tuple[dict[str, object], str]:
    """``(front matter dict, full text)``; the parser itself skips the front matter lines."""
    lines = text.splitlines()
    end = front_matter_end(lines)
    if end == 0:
        raise KBValidationError("missing YAML front matter")
    data = yaml.safe_load("\n".join(lines[1 : end - 1])) or {}
    if not isinstance(data, dict):
        raise KBValidationError("front matter must be a mapping")
    return data, text


def load_kb(kb_dir: Path, cfg: InternalKBConfig, principals: PrincipalsConfig) -> list[KBDocument]:
    """Read and validate every ``kb-*.md`` under ``kb_dir``.

    Raises:
        KBValidationError: on any per-file or cross-document inconsistency.
    """
    docs: list[KBDocument] = []
    for path in sorted(kb_dir.glob("kb-*.md")):
        data, text = split_front_matter(path.read_text(encoding="utf-8"))
        try:
            front = KBFrontMatter.model_validate(data)
        except ValueError as exc:
            raise KBValidationError(f"{path.name}: {exc}") from exc
        docs.append(KBDocument(front, text, path))
    validate_kb(docs, cfg, principals)
    return docs


def validate_kb(
    docs: list[KBDocument], cfg: InternalKBConfig, principals: PrincipalsConfig
) -> None:
    by_id: dict[str, KBDocument] = {}
    known_groups = set(principals.acl_groups)
    for doc in docs:
        f = doc.front
        if doc.path.name != f"{f.id}.md":
            raise KBValidationError(f"{doc.path.name}: file name must be {f.id}.md")
        if f.id in by_id:
            raise KBValidationError(f"duplicate id {f.id}")
        if f.doc_type not in cfg.doc_types:
            raise KBValidationError(f"{f.id}: unknown doc_type {f.doc_type!r}")
        if unknown := set(f.acl) - known_groups:
            raise KBValidationError(f"{f.id}: unknown ACL groups {sorted(unknown)}")
        by_id[f.id] = doc
    for doc in docs:
        f = doc.front
        if f.supersedes is None:
            continue
        old = by_id.get(f.supersedes)
        if old is None:
            raise KBValidationError(f"{f.id} supersedes missing {f.supersedes}")
        # Strictly increasing dates along supersedes edges also rule out cycles.
        if old.front.updated_at >= f.updated_at:
            raise KBValidationError(f"{f.id} must be newer than the {f.supersedes} it supersedes")


def superseded_by(docs: list[KBDocument]) -> dict[str, str]:
    """Map ``old id -> newer id`` along supersedes edges."""
    return {d.front.supersedes: d.front.id for d in docs if d.front.supersedes}


@dataclass
class ParsedKB:
    sections: list[Section]
    issues: list[ParseIssue]


def kb_doc_id(source: str, kb_id: str) -> str:
    return make_doc_id(source, f"{kb_id}.md")


def parse_kb(docs: list[KBDocument], cfg: InternalKBConfig, repo_root: Path) -> ParsedKB:
    """Sections of every KB doc; ACL, version, status and freshness come from front matter."""
    out = ParsedKB([], [])
    for doc in docs:
        f = doc.front
        meta = PageMeta(
            source=cfg.source,
            doc_type=f.doc_type,
            acl=tuple(f.acl),
            updated_at=datetime.combine(f.updated_at, time(0, 0), tzinfo=UTC),
            version=f.version,
            status=f.status,
            supersedes=kb_doc_id(cfg.source, f.supersedes) if f.supersedes else None,
        )
        try:
            provenance = doc.path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            provenance = doc.path.as_posix()
        page = parse_page(doc.body, rel_path=f"{f.id}.md", meta=meta, provenance_file=provenance)
        out.sections.extend(page.sections)
        out.issues.extend(page.issues)
    return out
