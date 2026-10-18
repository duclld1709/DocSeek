"""Corpus C: render selected docs to PDF, degrade them like scans, keep exact ground truth.

Pipeline (PROJECT_SPEC §3.3), per document:

1. Markdown (from processed sections, includes resolved) → HTML → PDF with PyMuPDF ``Story``
   (no browser or GTK). Every third document is laid out in two columns. At most
   ``max_pages`` pages per document.
2. Ground truth = the clean PDF's text layer per page (what is printed, in reading order),
   written to ``data/scans/ground_truth/<name>_p<N>.txt``; markdown tables of the source go to
   ``<name>.tables.json`` (header + rows).
3. Each page is rasterised at ``dpi``, rotated ±2°, blurred, noised and lowered in contrast
   (seeded per document), and the images are assembled into a PDF **without a text layer**:
   ``data/scans/pdf/<name>.pdf``.
4. ``data/scans/ground_truth/manifest.json`` records doc_id, ACL, metadata and pages per scan.
"""

from __future__ import annotations

import io
import json
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docseek.core.models import Section

CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.35; }
h1 { font-size: 16pt; } h2 { font-size: 13pt; } h3, h4 { font-size: 11pt; }
pre, code { font-family: monospace; font-size: 8.5pt; }
pre { white-space: pre-wrap; }
table { border-collapse: collapse; } td, th { border: 1px solid #444; padding: 2pt 4pt; }
"""
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")


@dataclass(frozen=True)
class ScanSpec:
    name: str
    doc_id: str
    two_columns: bool


def select_docs(sections: Sequence[Section], n_docs: int = 20, n_kb: int = 10) -> list[str]:
    """Docs pages with tables first, then pages with code spread over the corpus; KB docs with
    tables (SLA, error codes) first, then the earliest others."""
    by_doc: defaultdict[str, list[Section]] = defaultdict(list)
    for s in sections:
        by_doc[s.doc_id].append(s)

    def has_table(d: str) -> bool:
        return any(_TABLE_ROW.match(line) for s in by_doc[d] for line in s.text.splitlines())

    def has_code(d: str) -> bool:
        return any("```" in s.text for s in by_doc[d])

    docs = sorted(d for d in by_doc if d.startswith("fastapi-docs:"))
    tables = [d for d in docs if has_table(d)]
    code = [d for d in docs if has_code(d) and d not in tables]
    step = max(1, len(code) // max(1, n_docs - len(tables)))
    picked = (tables + code[::step])[:n_docs]
    kb = sorted(d for d in by_doc if d.startswith("internal-kb:"))
    kb_tables = [d for d in kb if has_table(d)]
    picked += (kb_tables + [d for d in kb if d not in kb_tables])[:n_kb]
    return picked


def doc_markdown(sections: Sequence[Section]) -> str:
    parts = []
    for s in sections:
        depth = max(1, min(4, len(s.section_path)))
        parts.append(f"{'#' * depth} {s.title}\n\n{s.text}\n")
    return "\n".join(parts)


def markdown_tables(md: str) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    rows: list[list[str]] = []
    for line in [*md.splitlines(), ""]:
        m = _TABLE_ROW.match(line)
        if m:
            cells = [c.strip() for c in m.group(1).split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                rows.append(cells)
        elif rows:
            tables.append({"header": rows[0], "rows": rows[1:]})
            rows = []
    return tables


def render_pdf(md: str, path: Path, *, two_columns: bool, max_pages: int) -> int:
    import markdown as md_lib
    import pymupdf

    html = md_lib.markdown(md, extensions=["tables", "fenced_code"])
    story = pymupdf.Story(html=html, user_css=CSS)
    writer = pymupdf.DocumentWriter(str(path))
    page_rect = pymupdf.paper_rect("a4")
    margin = 50
    content = pymupdf.Rect(margin, margin, page_rect.x1 - margin, page_rect.y1 - margin)
    if two_columns:
        mid = (content.x0 + content.x1) / 2
        cols = [
            pymupdf.Rect(content.x0, content.y0, mid - 10, content.y1),
            pymupdf.Rect(mid + 10, content.y0, content.x1, content.y1),
        ]
    else:
        cols = [content]
    pages, more = 0, 1
    while more and pages < max_pages:
        device = writer.begin_page(page_rect)
        for rect in cols:
            more, _ = story.place(rect)
            story.draw(device)
            if not more:
                break
        writer.end_page()
        pages += 1
    writer.close()
    return pages


def degrade(png: bytes, seed: int) -> bytes:
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    rng = random.Random(seed)
    img = Image.open(io.BytesIO(png)).convert("L")
    img = img.rotate(
        rng.uniform(-2, 2), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255
    )
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 0.9)))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 0.85))
    arr = np.asarray(img, dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0, rng.uniform(6, 12), arr.shape)
    out = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def make_scans(
    sections: Sequence[Section],
    out_dir: Path,
    *,
    dpi: int = 150,
    max_pages: int = 2,
    seed: int = 13,
) -> dict[str, Any]:
    import pymupdf

    by_doc: defaultdict[str, list[Section]] = defaultdict(list)
    for s in sections:
        by_doc[s.doc_id].append(s)
    gt_dir, pdf_dir, clean_dir = out_dir / "ground_truth", out_dir / "pdf", out_dir / "clean"
    for d in (gt_dir, pdf_dir, clean_dir):
        d.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"dpi": dpi, "seed": seed, "scans": []}
    for i, doc_id in enumerate(select_docs(sections)):
        secs = by_doc[doc_id]
        name = re.sub(r"[^a-z0-9]+", "_", doc_id.split(":", 1)[1].removesuffix(".md").lower())
        name = f"{doc_id.split(':')[0].split('-')[0]}_{name}"
        two_col = i % 3 == 2
        md = doc_markdown(secs)
        clean = clean_dir / f"{name}.pdf"
        n_pages = render_pdf(md, clean, two_columns=two_col, max_pages=max_pages)
        src = pymupdf.open(str(clean))
        scan = pymupdf.open()
        for p in range(n_pages):
            page = src[p]
            (gt_dir / f"{name}_p{p + 1}.txt").write_text(
                # Only what is visible on the page (wide lines may overflow the margin).
                page.get_text("text", clip=page.rect),
                encoding="utf-8",
                newline="\n",
            )
            png = degrade(page.get_pixmap(dpi=dpi).tobytes("png"), seed=seed * 1000 + i * 10 + p)
            out = scan.new_page(width=page.rect.width, height=page.rect.height)
            out.insert_image(out.rect, stream=png)
        scan.save(str(pdf_dir / f"{name}.pdf"), deflate=True)
        tables = markdown_tables(md)
        if tables:
            (gt_dir / f"{name}.tables.json").write_text(
                json.dumps(tables, indent=1), encoding="utf-8", newline="\n"
            )
        first = secs[0]
        manifest["scans"].append(
            {
                "name": name,
                "doc_id": doc_id,
                "pages": n_pages,
                "two_columns": two_col,
                "acl": list(first.acl),
                "doc_type": first.doc_type,
                "status": first.status,
                "version": first.version,
                "supersedes": first.supersedes,
                "updated_at": first.updated_at.isoformat() if first.updated_at else None,
                "title": first.section_path[0] if first.section_path else first.title,
                "n_tables": len(tables),
            }
        )
    (gt_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8", newline="\n"
    )
    return manifest
