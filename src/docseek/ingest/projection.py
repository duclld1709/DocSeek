"""Project the corpus into a markdown tree for filesystem-style navigation (M6).

``data/projected/<source>/<path>.md``: every section is a heading at its depth followed by
``<!-- section_id: ... -->`` and its text. ``_manifest.json`` maps each file to its ACL and
its sections' line numbers; a human-readable ``_index.md`` is written per directory. The FS
tools never serve the static ``_index.md`` or the manifest: they render a per-principal index
from the manifest, so restricted page titles never leak through an index.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from docseek.core.models import Section

MANIFEST = "_manifest.json"
INDEX = "_index.md"


def projected_path(doc_id: str) -> str:
    source, rel = doc_id.split(":", 1)
    return f"{source}/{rel}"


def render_page(sections: Sequence[Section]) -> tuple[str, list[dict[str, Any]]]:
    lines: list[str] = []
    toc: list[dict[str, Any]] = []
    for s in sections:
        depth = max(1, min(6, len(s.section_path)))
        toc.append(
            {
                "section_id": s.section_id,
                "title": s.title,
                "path": " > ".join(s.section_path),
                "line": len(lines) + 1,
            }
        )
        lines += [f"{'#' * depth} {s.title}", f"<!-- section_id: {s.section_id} -->", ""]
        if s.text.strip():
            lines += [*s.text.splitlines(), ""]
    return "\n".join(lines) + "\n", toc


def project(sections: Sequence[Section], out_dir: Path) -> dict[str, Any]:
    """Write the tree and manifest; returns the manifest. Fails if a page mixes ACLs."""
    by_doc: defaultdict[str, list[Section]] = defaultdict(list)
    for s in sections:
        by_doc[s.doc_id].append(s)
    files: dict[str, Any] = {}
    for doc_id, secs in sorted(by_doc.items()):
        acls = {tuple(sorted(s.acl)) for s in secs}
        if len(acls) != 1:
            raise ValueError(f"{doc_id}: sections with different ACLs cannot share one file")
        rel = projected_path(doc_id)
        text, toc = render_page(secs)
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        files[rel] = {
            "doc_id": doc_id,
            "acl": list(acls.pop()),
            "title": secs[0].section_path[0] if secs[0].section_path else secs[0].title,
            "status": secs[0].status,
            "sections": toc,
            "n_lines": text.count("\n"),
        }
    by_dir: defaultdict[str, list[str]] = defaultdict(list)
    for rel in files:
        by_dir[str(Path(rel).parent.as_posix())].append(rel)
    for d, rels in by_dir.items():
        (out_dir / d / INDEX).write_text(
            render_index(d, rels, files), encoding="utf-8", newline="\n"
        )
    manifest = {"files": files}
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return manifest


def render_index(directory: str, rels: Sequence[str], files: dict[str, Any]) -> str:
    out = [f"# Index of {directory}/", ""]
    for rel in sorted(rels):
        meta = files[rel]
        status = f" ({meta['status']})" if meta["status"] != "active" else ""
        out.append(f"## {Path(rel).name} - {meta['title']}{status}")
        for t in meta["sections"]:
            out.append(f"- L{t['line']}: {t['path']}  `{t['section_id']}`")
        out.append("")
    return "\n".join(out) + "\n"
