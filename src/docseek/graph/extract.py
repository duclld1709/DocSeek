"""Build graph nodes and edges from the processed corpus, ``docs_src`` and release notes.

Sources, cheapest first:

- **Structure**: Page/Section/PART_OF from ``sections.jsonl``; LINKS_TO from ``Section.links``;
  SHOWS from ``Section.code_refs``; KBDoc, SUPERSEDES from the KB front matter.
- **AST of ``docs_src/*.py``**: names imported from ``fastapi``/``starlette``/``pydantic`` are the
  API vocabulary (APIElement); a file USES every imported name it references.
- **Rules**: a section MENTIONS an API element when the name appears in backticks (``Depends``,
  ``fastapi.Depends()``) or, for compound identifiers (``BackgroundTasks``, ``response_model``),
  anywhere as a whole word. Release-note headings ``## 0.143.4 (2026-10-17)`` become Version
  nodes; API names in backticks under them give CHANGED_IN.
- **LLM (optional, cached)**: up to N key concepts per page; a section MENTIONS a concept when
  the normalised concept name occurs in its title or text.

Entity resolution normalises names: API names drop module prefixes, ``()`` and ``@``; concept
names are lower-cased, whitespace-collapsed and naively singularised, then merged.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docseek.core.models import Section

API_MODULES = ("fastapi", "starlette", "pydantic")
_BACKTICK = re.compile(r"`([^`\n]{1,80})`")
_VERSION = re.compile(r"^##\s+(\d+\.\d+\.\d+)\s*(?:\((\d{4}-\d{2}-\d{2})\))?\s*$")

Edge = tuple[str, str, str, str, str]  # (src_label, src_key, rel, dst_label, dst_key)


@dataclass
class GraphData:
    nodes: dict[str, dict[str, dict[str, Any]]] = field(default_factory=lambda: defaultdict(dict))
    edges: set[Edge] = field(default_factory=set)

    def node(self, label: str, key: str, **props: Any) -> None:
        self.nodes[label].setdefault(key, {}).update(props)

    def edge(self, src: tuple[str, str], rel: str, dst: tuple[str, str]) -> None:
        self.edges.add((src[0], src[1], rel, dst[0], dst[1]))

    def counts(self) -> dict[str, int]:
        out = {label: len(v) for label, v in self.nodes.items()}
        for e in self.edges:
            k = f"{e[0]}-{e[2]}->{e[3]}"
            out[k] = out.get(k, 0) + 1
        return out


def normalize_api(name: str) -> str:
    """``fastapi.Depends()`` → ``Depends``; ``@app.get`` stays ``app.get`` (not in vocab)."""
    name = name.strip().lstrip("@").removesuffix("()")
    name = re.sub(r"\(.*\)$", "", name)
    for mod in API_MODULES:
        if name.startswith(mod + "."):
            name = name.rsplit(".", 1)[-1]
    return name


def normalize_concept(name: str) -> str:
    words = re.sub(r"[^a-z0-9+#./ -]", " ", name.lower()).split()
    out = []
    for w in words:
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        out.append(w)
    return " ".join(out)


def _is_compound(name: str) -> bool:
    return "_" in name or bool(re.search(r"[a-z][A-Z]|[A-Z]{2,}[a-z]", name))


@dataclass(frozen=True)
class CodeFile:
    path: str
    imports: dict[str, str]  # name -> module
    used: frozenset[str]


def parse_code_file(path: str, source: str) -> CodeFile | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.module.split(".")[0] not in API_MODULES:
            continue
        for alias in node.names:
            if alias.name != "*":
                imports[alias.asname or alias.name] = node.module
    referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    referenced |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    return CodeFile(path, imports, frozenset(n for n in imports if n in referenced))


def api_mentions(text: str, vocab: Mapping[str, str]) -> set[str]:
    """API names (canonical case) mentioned in ``text``; ``vocab`` maps lower → canonical."""
    found = {vocab[k] for m in _BACKTICK.findall(text) if (k := normalize_api(m).lower()) in vocab}
    for name in vocab.values():
        if _is_compound(name) and re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", text):
            found.add(name)
    return found


def concept_mentions(text: str, concepts: Iterable[str]) -> set[str]:
    norm = " " + normalize_concept(text) + " "
    return {c for c in concepts if f" {c} " in norm}


def parse_release_notes(
    text: str, vocab: Mapping[str, str]
) -> list[tuple[str, str | None, set[str]]]:
    """``[(version, date, api names in backticks under that heading)]``."""
    out: list[tuple[str, str | None, set[str]]] = []
    current: tuple[str, str | None] | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = _VERSION.match(line)
        if m or line.startswith("## "):
            if current:
                out.append((*current, api_mentions("\n".join(buf), vocab)))
            current = (m.group(1), m.group(2)) if m else None
            buf = []
        elif current:
            buf.append(line)
    if current:
        out.append((*current, api_mentions("\n".join(buf), vocab)))
    return out


# Words the concept LLM sometimes returns that carry no topic (Python keywords, glue words).
_STOP = (
    "with from import as def class async await return yield python fastapi str int dict list "
    "none true false self app code example file data value type function"
)
CONCEPT_STOPLIST = frozenset(_STOP.split())

ConceptFn = Callable[[str, str], list[str]]
"""(page title, page text) -> concept names."""


def build_graph(
    sections: Sequence[Section],
    code_files: Sequence[CodeFile],
    *,
    release_notes: str | None = None,
    concept_fn: ConceptFn | None = None,
    max_concept_pages: int = 0,
    max_concept_df: float = 0.05,
) -> GraphData:
    g = GraphData()
    vocab: dict[str, str] = {}
    for f in code_files:
        g.node("CodeExample", f.path, path=f.path)
        for name in f.imports:
            vocab.setdefault(name.lower(), name)
            kind = "class" if name[:1].isupper() else "function"
            g.node("APIElement", name, name=name, kind=kind, module=f.imports[name])
        for name in f.used:
            g.edge(("CodeExample", f.path), "USES", ("APIElement", name))

    by_doc: defaultdict[str, list[Section]] = defaultdict(list)
    for s in sections:
        by_doc[s.doc_id].append(s)
    page_ids = set(by_doc)
    for doc_id, secs in by_doc.items():
        first = secs[0]
        acl = sorted({a for s in secs for a in s.acl})
        g.node(
            "Page",
            doc_id,
            doc_id=doc_id,
            title=first.section_path[0] if first.section_path else first.title,
            doc_type=first.doc_type,
            source=first.source,
            acl=acl,
        )
        if first.source == "internal-kb":
            g.node(
                "KBDoc",
                doc_id,
                doc_id=doc_id,
                acl=list(first.acl),
                status=first.status,
                version=first.version,
                updated_at=first.updated_at.isoformat() if first.updated_at else None,
            )
            if first.supersedes and first.supersedes in page_ids:
                g.edge(("KBDoc", doc_id), "SUPERSEDES", ("KBDoc", first.supersedes))

    for s in sections:
        g.node(
            "Section",
            s.section_id,
            section_id=s.section_id,
            doc_id=s.doc_id,
            title=s.title,
            path=" > ".join(s.section_path),
            acl=list(s.acl),
            status=s.status,
        )
        g.edge(("Section", s.section_id), "PART_OF", ("Page", s.doc_id))
        for link in s.links:
            target = link.split("#", 1)[0]
            if target in page_ids and target != s.doc_id:
                g.edge(("Page", s.doc_id), "LINKS_TO", ("Page", target))
        for ref in s.code_refs:
            if ref in g.nodes["CodeExample"]:
                g.edge(("Section", s.section_id), "SHOWS", ("CodeExample", ref))
        for name in api_mentions(f"{s.title}\n{s.text}", vocab):
            g.edge(("Section", s.section_id), "MENTIONS", ("APIElement", name))
            if s.source == "internal-kb":
                g.edge(("KBDoc", s.doc_id), "REFERENCES", ("APIElement", name))

    if release_notes:
        for version, date, apis in parse_release_notes(release_notes, vocab):
            g.node("Version", version, name=version, date=date)
            for name in apis:
                g.edge(("APIElement", name), "CHANGED_IN", ("Version", version))

    if concept_fn is not None and max_concept_pages > 0:
        concepts: set[str] = set()
        for doc_id in sorted(by_doc)[:max_concept_pages]:
            secs = by_doc[doc_id]
            text = "\n\n".join(s.text for s in secs)[:4000]
            title = secs[0].section_path[0] if secs[0].section_path else secs[0].title
            for raw in concept_fn(title, text):
                c = normalize_concept(raw)
                if 3 <= len(c) <= 40:
                    concepts.add(c)
        # Entity resolution / pruning: drop glue words, concepts that are API names, and
        # concepts so common (> max_concept_df of sections) that they cannot discriminate.
        concepts = {
            c
            for c in concepts
            if c not in CONCEPT_STOPLIST and c.replace(" ", "") not in vocab and len(c) >= 4
        }
        mentions = {
            s.section_id: concept_mentions(f"{s.title}\n{s.text}", concepts) for s in sections
        }
        df: defaultdict[str, int] = defaultdict(int)
        for found in mentions.values():
            for c in found:
                df[c] += 1
        limit = max_concept_df * len(sections)
        kept = {c for c in concepts if 0 < df[c] <= limit}
        for c in kept:
            g.node("Concept", c, name=c, df=df[c])
        for s in sections:
            for c in mentions[s.section_id] & kept:
                g.edge(("Section", s.section_id), "MENTIONS", ("Concept", c))
                if s.source == "internal-kb":
                    g.edge(("KBDoc", s.doc_id), "REFERENCES", ("Concept", c))
    return g


def load_code_files(root: Path, repo_root: Path) -> list[CodeFile]:
    """Parse every ``*.py`` under ``root``; paths are relative to ``repo_root``."""
    out: list[CodeFile] = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(repo_root).as_posix()
        parsed = parse_code_file(rel, p.read_text(encoding="utf-8", errors="replace"))
        if parsed is not None:
            out.append(parsed)
    return out
