"""Corpus C scans as retrievable pages, and the corpus variant that swaps originals for them.

A scanned PDF page becomes one *page section* ``scan:<name>.pdf#page-<n>`` with
``modality="scan"``, ``provenance.page`` and the original document's ACL/metadata, so an answer
can cite "page X of file Y". Its text is OCR output (never the ground truth).

Which original sections a page shows is recovered from the **ground truth** text: section
titles are rendered as heading lines in document order, so each title is searched for from
the previous title's position on; a section spans the pages from its heading to the next
rendered heading. Sections whose title is never found after the last match were cut off by
the page limit and stay in the text corpus.

In the variant corpus the rendered sections are removed (the answer then exists *only* as a
scan), and a golden query that pointed at a removed section points at its pages instead, with
the same grade (``remap_relevance``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from docseek.core.ids import make_chunk_id
from docseek.core.models import Chunk, Section

SCAN_STRATEGY = "scan"
SCAN_SOURCE = "scan"


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[`*_]", "", text).split()).lower()


def page_section_id(name: str, page: int) -> str:
    return f"{SCAN_SOURCE}:{name}.pdf#page-{page}"


def section_pages(sections: Sequence[Section], gt_pages: Sequence[str]) -> dict[str, list[int]]:
    """``section_id -> 1-based pages`` for the sections of one doc that were rendered."""
    texts = [_norm(p) for p in gt_pages]
    full = " ".join(texts)
    starts, offset = [], 0
    for t in texts:  # char offset where each page starts in ``full``
        starts.append(offset)
        offset += len(t) + 1

    def page_at(pos: int) -> int:
        return max(i for i, s in enumerate(starts) if s <= pos) + 1

    found: list[tuple[str, int]] = []  # (section_id, heading position)
    cursor = 0
    for s in sections:
        pos = full.find(_norm(s.title), cursor)
        if pos >= 0:
            found.append((s.section_id, pos))
            cursor = pos + 1
    out: dict[str, list[int]] = {}
    ids = [s.section_id for s in sections]
    for i, (sid, pos) in enumerate(found):
        end = found[i + 1][1] - 1 if i + 1 < len(found) else len(full) - 1
        first, last = page_at(pos), page_at(max(pos, end))
        out[sid] = list(range(first, last + 1))
        # Untitled-match sections between two found headings sit on the same pages.
        if i + 1 < len(found):
            for between in ids[ids.index(sid) + 1 : ids.index(found[i + 1][0])]:
                out[between] = list(range(first, last + 1))
    return out


def scan_chunks(scan: Mapping[str, Any], page_texts: Sequence[str], engine: str) -> list[Chunk]:
    """One chunk per scanned page, metadata from the manifest entry of the original doc."""
    name = scan["name"]
    out = []
    for page, text in enumerate(page_texts, start=1):
        sid = page_section_id(name, page)
        title = f"{scan['title']} (scan {name}.pdf, page {page})"
        meta = {
            "section_id": sid,
            "doc_id": sid.split("#", 1)[0],
            "source": SCAN_SOURCE,
            "title": title,
            "section_path": scan["title"],
            "doc_type": scan["doc_type"],
            "modality": "scan",
            "acl": list(scan["acl"]),
            "status": scan["status"],
            "version": scan["version"],
            "supersedes": scan["supersedes"],
            "updated_at": scan["updated_at"],
            "original_doc_id": scan["doc_id"],
            "file": f"data/scans/pdf/{name}.pdf",
            "page": page,
            "ocr_engine": engine,
        }
        body = text.strip() or "(no text recognised)"
        out.append(
            Chunk(
                chunk_id=make_chunk_id(sid, SCAN_STRATEGY, 0),
                section_id=sid,
                strategy=SCAN_STRATEGY,
                text=body,
                text_for_embedding=f"{title}\n{body}",
                metadata=meta,
                n_tokens=len(body.split()),
            )
        )
    return out


def remap_relevance(
    relevant: Mapping[str, int], replaced: Mapping[str, Sequence[str]]
) -> dict[str, int]:
    """Point relevance at page sections for every replaced section (grade = max over sources)."""
    out: dict[str, int] = {}
    for sid, grade in relevant.items():
        for target in replaced.get(sid, [sid]):
            out[target] = max(grade, out.get(target, 0))
    return out
