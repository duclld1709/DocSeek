"""Split FastAPI-flavoured markdown into heading-delimited sections (PROJECT_SPEC §3.1).

The parser is line based on purpose: it keeps exact line numbers for provenance and copes
with the mkdocs dialect FastAPI uses (``{ #anchor }`` heading suffixes, ``{* ... *}``
code includes, ``///`` admonitions), which a CommonMark parser would not understand.

Rules:

- A heading is a line starting with 1-6 ``#`` and a space, **outside fenced code**.
- ``## Title { #anchor }`` gives ``title="Title"`` and ``anchor="anchor"``; headings
  without an explicit anchor get a slug like mkdocs' default ``toc`` extension would make.
- Anchors are unique within a page (``_1``, ``_2``... suffixes, as mkdocs does).
- Text before the first heading is prepended to the first section.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

from docseek.core.ids import make_doc_id, make_section_id
from docseek.core.models import Provenance, Section

FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})[ \t]+(?P<body>.+?)[ \t]*$")
ANCHOR_SUFFIX_RE = re.compile(r"[ \t]*\{[ \t]*#(?P<anchor>[^\s}]+)[ \t]*\}[ \t]*$")
_MD_LINK_RE = re.compile(r"!?\[(?P<text>[^\]]*)\]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


@dataclass(frozen=True)
class PageMeta:
    """Page-level metadata copied onto every section of the page."""

    source: str
    doc_type: str
    acl: tuple[str, ...]
    updated_at: datetime | None = None
    version: int | None = None
    status: str = "active"
    supersedes: str | None = None


@dataclass
class _RawSection:
    level: int
    title: str
    anchor: str
    path: list[str]
    line_start: int  # 1-based, the heading line
    line_end: int
    body: list[tuple[int, str]] = field(default_factory=list)  # (line number, text)


def slugify(text: str) -> str:
    """Slug like mkdocs/Python-Markdown ``toc``: ascii, lowercase, ``-`` between words."""
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value).strip("-") or "section"


def clean_title(raw: str) -> str:
    """Heading text without anchor suffix, link targets, backticks or HTML tags."""
    title = ANCHOR_SUFFIX_RE.sub("", raw)
    title = _MD_LINK_RE.sub(lambda m: m["text"], title)
    title = _HTML_TAG_RE.sub("", title).replace("`", "")
    return " ".join(title.split())


def front_matter_end(lines: list[str]) -> int:
    """Number of leading lines taken by a YAML front matter block (0 if none)."""
    if not lines or lines[0].strip() != "---":
        return 0
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            return i + 1
    return 0


class FenceTracker:
    """Tracks whether a line is inside fenced code (```` ``` ```` or ``~~~``)."""

    def __init__(self) -> None:
        self._open: str | None = None  # the opening fence, e.g. "```"
        self.info = ""

    @property
    def inside(self) -> bool:
        return self._open is not None

    def feed(self, line: str) -> bool:
        """Consume a line; return True if it is a fence delimiter or inside a fence."""
        m = FENCE_RE.match(line)
        if self._open is None:
            if m:
                self._open = m["fence"]
                self.info = m["info"].strip()
                return True
            return False
        stripped = line.strip()
        if (
            m
            and m["fence"][0] == self._open[0]
            and len(m["fence"]) >= len(self._open)
            and not m["info"].strip()
            and stripped == m["fence"]
        ):
            self._open = None
            self.info = ""
        return True


def page_path_prefix(rel_path: str) -> list[str]:
    """Breadcrumb from the page's folders, e.g. ``tutorial/dependencies/x.md`` -> Tutorial >
    Dependencies. For ``index.md`` the last folder *is* the page, so it is left out."""
    parts = rel_path.split("/")[:-1]
    if rel_path.endswith("/index.md") or rel_path == "index.md":
        parts = parts[:-1]
    return [p.replace("-", " ").replace("_", " ").title() for p in parts]


def split_sections(lines: list[str], rel_path: str) -> list[_RawSection]:
    """Split page lines into raw sections with titles, unique anchors and line ranges."""
    start = front_matter_end(lines)
    fences = FenceTracker()
    prefix = page_path_prefix(rel_path)
    sections: list[_RawSection] = []
    preamble: list[tuple[int, str]] = []
    stack: list[_RawSection] = []
    used: dict[str, int] = {}

    for idx in range(start, len(lines)):
        lineno, line = idx + 1, lines[idx]
        if fences.feed(line):
            (sections[-1].body if sections else preamble).append((lineno, line))
            continue
        m = HEADING_RE.match(line)
        if not m:
            (sections[-1].body if sections else preamble).append((lineno, line))
            continue
        level = len(m["hashes"])
        title = clean_title(m["body"])
        explicit = ANCHOR_SUFFIX_RE.search(m["body"])
        anchor = _unique(explicit["anchor"] if explicit else slugify(title), used)
        while stack and stack[-1].level >= level:
            stack.pop()
        path = [*prefix, *(s.title for s in stack), title]
        section = _RawSection(level, title, anchor, path, lineno, lineno)
        sections.append(section)
        stack.append(section)

    if not sections:
        stem = rel_path.rsplit("/", 1)[-1].removesuffix(".md")
        title = stem.replace("-", " ").replace("_", " ").title()
        first_line = start + 1 if start < len(lines) else 1
        sections.append(
            _RawSection(1, title, _unique(slugify(stem), used), [*prefix, title], first_line, 1)
        )
    if any(text.strip() for _, text in preamble):
        sections[0].body[:0] = preamble
        sections[0].line_start = min(sections[0].line_start, preamble[0][0])

    for i, section in enumerate(sections):
        nxt = sections[i + 1].line_start - 1 if i + 1 < len(sections) else len(lines)
        section.line_end = max(section.line_start, nxt)
    return sections


def _unique(anchor: str, used: dict[str, int]) -> str:
    if anchor not in used:
        used[anchor] = 0
        return anchor
    while True:
        used[anchor] += 1
        candidate = f"{anchor}_{used[anchor]}"
        if candidate not in used:
            used[candidate] = 0
            return candidate


def parse_page(
    text: str,
    *,
    rel_path: str,
    meta: PageMeta,
    provenance_file: str,
) -> list[Section]:
    """Parse one markdown page into sections.

    Args:
        text: Page content.
        rel_path: Path relative to the source root (becomes part of ``doc_id``).
        meta: Page-level metadata (source, doc_type, ACL, freshness).
        provenance_file: Path recorded in provenance (relative to the source repo).
    """
    lines = text.splitlines()
    doc_id = make_doc_id(meta.source, rel_path)
    sections: list[Section] = []
    for raw in split_sections(lines, rel_path):
        body = "\n".join(line for _, line in raw.body)
        sections.append(
            Section(
                section_id=make_section_id(doc_id, raw.anchor),
                doc_id=doc_id,
                source=meta.source,
                title=raw.title,
                anchor=raw.anchor,
                section_path=raw.path,
                text=_tidy(body),
                doc_type=meta.doc_type,
                acl=list(meta.acl),
                updated_at=meta.updated_at,
                version=meta.version,
                status=meta.status,
                supersedes=meta.supersedes,
                provenance=Provenance(
                    file=provenance_file, line_start=raw.line_start, line_end=raw.line_end
                ),
            )
        )
    return sections


def _tidy(text: str) -> str:
    """Strip trailing spaces and collapse runs of blank lines."""
    lines = [line.rstrip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
