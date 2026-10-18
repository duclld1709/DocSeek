"""``docseek ingest``: parse every corpus into sections, chunk them, write JSONL + stats.

Outputs (``data/processed/``):

- ``sections.jsonl``: one ``Section`` per line, the unit relevance is judged at.
- ``chunks.jsonl``: one ``Chunk`` per line, the unit that gets indexed.
- ``stats.json``: corpus statistics, parse issues and the input hash.

The run is skipped when the input hash (corpus commit, KB files, chunking settings, token
counter and the ingest code itself) matches the previous run, so re-running is free.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from docseek.core.config import CorpusConfig, PrincipalsConfig
from docseek.core.ids import doc_id_of_section
from docseek.core.logging import get_logger
from docseek.core.models import Chunk, Section
from docseek.ingest.chunking import ChunkingConfig, chunk_sections
from docseek.ingest.includes import CodeIncludeResolver, find_unresolved_includes
from docseek.ingest.internal_kb import load_kb, parse_kb
from docseek.ingest.markdown_parser import PageMeta, ParseIssue, parse_page
from docseek.ingest.metadata import PUBLIC_ACL, doc_type_for, git_updated_at, is_excluded
from docseek.ingest.tokens import TokenCounter

log = get_logger(__name__)

PIPELINE_VERSION = 1


@dataclass
class IngestResult:
    sections: list[Section]
    chunks: list[Chunk]
    issues: list[tuple[str, ParseIssue]] = field(default_factory=list)  # (doc_id, issue)
    link_stats: dict[str, int] = field(default_factory=dict)


def parse_fastapi(corpus: CorpusConfig, *, use_git_dates: bool = True) -> IngestResult:
    """Sections of every indexable FastAPI page at the pinned checkout."""
    cfg = corpus.fastapi
    repo = cfg.dest
    docs_root = repo / cfg.docs_root
    if not docs_root.is_dir():
        raise FileNotFoundError(f"{docs_root} not found; run `docseek data fetch` first")
    resolver = CodeIncludeResolver(repo_root=repo, mkdocs_dir=docs_root.parent)
    dates = git_updated_at(repo, cfg.docs_root) if use_git_dates else {}
    out = IngestResult([], [])
    for path in sorted(docs_root.rglob("*.md")):
        rel = path.relative_to(docs_root).as_posix()
        if is_excluded(rel, cfg):
            continue
        repo_rel = path.relative_to(repo).as_posix()
        meta = PageMeta(
            source=cfg.source,
            doc_type=doc_type_for(rel),
            acl=PUBLIC_ACL,
            updated_at=dates.get(repo_rel),
        )
        page = parse_page(
            path.read_text(encoding="utf-8"),
            rel_path=rel,
            meta=meta,
            provenance_file=repo_rel,
            includes=resolver,
        )
        out.sections.extend(page.sections)
        out.issues.extend((page.doc_id, i) for i in page.issues)
    return out


def resolve_links(sections: list[Section]) -> tuple[list[Section], dict[str, int]]:
    """Keep links whose target exists; a missing anchor falls back to its document.

    Returns the updated sections and counts of ``ok``, ``anchor_fallback`` and ``dangling``.
    """
    section_ids = {s.section_id for s in sections}
    doc_ids = {s.doc_id for s in sections}
    counts: Counter[str] = Counter()
    out: list[Section] = []
    for s in sections:
        links: list[str] = []
        for target in s.links:
            if target in section_ids or ("#" not in target and target in doc_ids):
                resolved: str | None = target
                counts["ok"] += 1
            elif "#" in target and doc_id_of_section(target) in doc_ids:
                resolved = doc_id_of_section(target)
                counts["anchor_fallback"] += 1
            else:
                resolved = None
                counts["dangling"] += 1
            if resolved and resolved != s.section_id and resolved not in links:
                links.append(resolved)
        out.append(s.model_copy(update={"links": links}) if links != s.links else s)
    return out, {"ok": 0, "anchor_fallback": 0, "dangling": 0, **counts}


def run_pipeline(
    corpus: CorpusConfig,
    principals: PrincipalsConfig,
    counter: TokenCounter,
    chunking: ChunkingConfig,
    *,
    repo_root: Path,
    use_git_dates: bool = True,
) -> IngestResult:
    """Parse Corpus A and (if present) Corpus B, resolve links and chunk."""
    result = parse_fastapi(corpus, use_git_dates=use_git_dates)
    kb_dir = corpus.internal_kb.dir
    if any(kb_dir.glob("kb-*.md")):
        kb = parse_kb(
            load_kb(kb_dir, corpus.internal_kb, principals), corpus.internal_kb, repo_root
        )
        result.sections.extend(kb.sections)
        result.issues.extend(("internal-kb", i) for i in kb.issues)
    else:
        log.warning("ingest.no_internal_kb", dir=str(kb_dir))
    result.sections, result.link_stats = resolve_links(result.sections)
    result.chunks = chunk_sections(result.sections, counter, chunking)
    return result


def corpus_stats(result: IngestResult, counter: TokenCounter) -> dict[str, Any]:
    """Numbers for docs/results: per source/doc_type/ACL, sizes, issues and links."""
    secs, chunks = result.sections, result.chunks
    sec_tokens = [counter.count(s.text) for s in secs if s.text.strip()]
    chunk_tokens = [c.n_tokens for c in chunks]
    by_source: dict[str, dict[str, int]] = {}
    for src in sorted({s.source for s in secs}):
        ss = [s for s in secs if s.source == src]
        by_source[src] = {
            "documents": len({s.doc_id for s in ss}),
            "sections": len(ss),
            "chunks": sum(1 for c in chunks if c.metadata["source"] == src),
        }
    return {
        "documents": len({s.doc_id for s in secs}),
        "sections": len(secs),
        "empty_sections": sum(1 for s in secs if not s.text.strip()),
        "chunks": len(chunks),
        "sections_split": sum(1 for n in Counter(c.section_id for c in chunks).values() if n > 1),
        "by_source": by_source,
        "by_doc_type": dict(sorted(Counter(s.doc_type for s in secs).items())),
        "by_acl": dict(sorted(Counter(g for s in secs for g in s.acl).items())),
        "by_status": dict(sorted(Counter(s.status for s in secs).items())),
        "section_tokens": _describe(sec_tokens),
        "chunk_tokens": _describe(chunk_tokens),
        "code_refs": len({r for s in secs for r in s.code_refs}),
        "links": result.link_stats,
        "unresolved_include_syntax": sum(len(find_unresolved_includes(s.text)) for s in secs),
        "issues": dict(sorted(Counter(i.kind for _, i in result.issues).items())),
        "issue_details": [
            {"doc_id": d, "kind": i.kind, "line": i.line, "detail": i.detail}
            for d, i in result.issues
        ],
        "token_counter": counter.name,
    }


def _describe(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 1),
    }


def input_hash(
    corpus: CorpusConfig, chunking: ChunkingConfig, counter: TokenCounter, *, commit: str | None
) -> str:
    """Hash of everything the outputs depend on."""
    h = hashlib.sha256()
    h.update(json.dumps([PIPELINE_VERSION, commit, chunking.__dict__, counter.name]).encode())
    h.update(corpus.model_dump_json().encode())
    for path in sorted(corpus.internal_kb.dir.glob("kb-*.md")):
        h.update(path.name.encode() + path.read_bytes())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode() + path.read_bytes())
    return h.hexdigest()


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    with path.open(encoding="utf-8") as fh:
        return [model.model_validate_json(line) for line in fh if line.strip()]


def is_up_to_date(out_dir: Path, digest: str) -> bool:
    stats = out_dir / "stats.json"
    if not (stats.is_file() and (out_dir / "sections.jsonl").is_file()):
        return False
    if not (out_dir / "chunks.jsonl").is_file():
        return False
    return bool(json.loads(stats.read_text(encoding="utf-8")).get("input_hash") == digest)
