"""Page metadata for Corpus A: doc_type, freshness from git, ACL and exclusions (§3.1)."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from docseek.core.config import FastAPICorpusConfig

DOC_TYPES = ("tutorial", "advanced", "how-to", "reference", "deployment", "learn", "about")
OTHER = "other"
PUBLIC_ACL = ("public",)


def doc_type_for(rel_path: str) -> str:
    """``doc_type`` from the top folder (relative to docs root); root pages are ``other``."""
    top, sep, _ = rel_path.partition("/")
    return top if sep and top in DOC_TYPES else OTHER


def is_excluded(rel_path: str, cfg: FastAPICorpusConfig) -> bool:
    """Whether a page (path relative to docs root) is excluded from indexing."""
    parts = rel_path.split("/")
    return rel_path in cfg.exclude_files or any(p in cfg.exclude_dirs for p in parts[:-1])


def git_updated_at(repo: Path, pathspec: str) -> dict[str, datetime]:
    """Last commit time (``%cI``) of every file under ``pathspec``, keyed by repo-relative path.

    One ``git log`` pass instead of one ``git log -1`` per file; the first time a path shows
    up in the (newest-first) log is its latest change. ``--no-renames`` avoids content
    diffs, which would make git fetch blobs in a blobless clone.

    Raises:
        RuntimeError: if git fails (for example ``repo`` is not a repository).
    """
    proc = subprocess.run(
        [
            "git",
            "-c",
            "core.quotepath=false",
            "log",
            "--no-renames",
            "--format=%x00%cI",
            "--name-only",
            "HEAD",
            "--",
            pathspec,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git log failed in {repo}: {proc.stderr.strip()}")
    updated: dict[str, datetime] = {}
    for record in proc.stdout.split("\0")[1:]:
        date_line, _, names = record.partition("\n")
        when = datetime.fromisoformat(date_line.strip())
        for name in names.split("\n"):
            name = name.strip()
            if name and name not in updated:
                updated[name] = when
    return updated
