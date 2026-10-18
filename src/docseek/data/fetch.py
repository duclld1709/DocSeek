"""Fetch Corpus A (FastAPI docs) at a pinned commit and write a manifest.

The clone is blobless (``--filter=blob:none``) and sparse: only ``sparse_paths`` are
checked out, but the full commit history is kept so per-file ``updated_at`` can later be
read with ``git log -1 -- <file>``. Line-ending conversion is disabled so file hashes in the
manifest are identical on every OS.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from docseek.core.config import FastAPICorpusConfig
from docseek.core.logging import get_logger

log = get_logger(__name__)

MANIFEST_SCHEMA_VERSION = 1
# Applied to every git call: keep bytes as committed and never prompt for credentials.
_GIT_CONFIG = ("-c", "core.autocrlf=false", "-c", "core.eol=lf", "-c", "core.symlinks=false")


class FetchError(RuntimeError):
    """Raised when the corpus cannot be fetched at the pinned commit."""


class ManifestFile(BaseModel):
    path: str
    sha256: str
    size: int


class Manifest(BaseModel):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    source: str
    repo_url: str
    commit: str
    commit_date: str
    fetched_at: str
    sparse_paths: list[str]
    file_count: int
    counts_by_extension: dict[str, int]
    counts_by_path: dict[str, int]
    files: list[ManifestFile] = Field(default_factory=list)


class FetchResult(BaseModel):
    manifest: Manifest
    cloned: bool


def _git(*args: str, cwd: Path | None = None) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    proc = subprocess.run(
        ["git", *_GIT_CONFIG, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise FetchError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _head(dest: Path) -> str | None:
    if not (dest / ".git").exists():
        return None
    try:
        return _git("rev-parse", "HEAD", cwd=dest).strip()
    except FetchError:
        return None


def _rmtree(path: Path) -> None:
    def _on_error(func: Any, p: str, _exc: BaseException) -> None:
        os.chmod(p, stat.S_IWRITE)  # git marks pack files read-only on Windows
        func(p)

    shutil.rmtree(path, onexc=_on_error)


def _clone(cfg: FastAPICorpusConfig, dest: Path) -> None:
    if dest.exists():
        _rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("git_clone", repo=cfg.repo_url, dest=str(dest))
    _git("clone", "--filter=blob:none", "--no-checkout", "--sparse", cfg.repo_url, str(dest))
    for key, value in (("core.autocrlf", "false"), ("core.eol", "lf")):
        _git("config", key, value, cwd=dest)
    _git("sparse-checkout", "set", "--cone", *cfg.sparse_paths, cwd=dest)
    try:
        _git("cat-file", "-e", f"{cfg.commit}^{{commit}}", cwd=dest)
    except FetchError:
        # Not reachable from the default branch: fetch the pinned commit explicitly.
        _git("fetch", "--filter=blob:none", "origin", cfg.commit, cwd=dest)
    _git("checkout", "--detach", cfg.commit, cwd=dest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(cfg: FastAPICorpusConfig, dest: Path) -> Manifest:
    """Hash every tracked file under the sparse paths of an already checked-out clone."""
    listed = _git("ls-files", "-z", "--", *cfg.sparse_paths, cwd=dest)
    paths = sorted(p for p in listed.split("\0") if p)
    files = [
        ManifestFile(path=p, sha256=_sha256(dest / p), size=(dest / p).stat().st_size)
        for p in paths
    ]
    by_ext = Counter(Path(p).suffix.lower() or "<none>" for p in paths)
    by_path = {
        sp: sum(1 for p in paths if p == sp or p.startswith(f"{sp.rstrip('/')}/"))
        for sp in cfg.sparse_paths
    }
    return Manifest(
        source=cfg.source,
        repo_url=cfg.repo_url,
        commit=_git("rev-parse", "HEAD", cwd=dest).strip(),
        commit_date=_git("show", "-s", "--format=%cI", "HEAD", cwd=dest).strip(),
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        sparse_paths=list(cfg.sparse_paths),
        file_count=len(files),
        counts_by_extension=dict(sorted(by_ext.items())),
        counts_by_path=by_path,
        files=files,
    )


def write_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((manifest.model_dump_json(indent=2) + "\n").encode("utf-8"))


def fetch_fastapi_docs(
    cfg: FastAPICorpusConfig, manifest_path: Path, *, force: bool = False
) -> FetchResult:
    """Ensure ``cfg.dest`` is a sparse clone at ``cfg.commit`` and (re)write the manifest.

    Idempotent: an existing clone already at the pinned commit is reused unless ``force``.

    Raises:
        FetchError: if git fails or HEAD does not end up at the pinned commit.
    """
    dest = cfg.dest
    cloned = force or _head(dest) != cfg.commit
    if cloned:
        _clone(cfg, dest)
    else:
        log.info("git_clone_skipped", dest=str(dest), commit=cfg.commit)

    head = _head(dest)
    if head != cfg.commit:
        raise FetchError(f"HEAD is {head}, expected pinned commit {cfg.commit}")

    manifest = build_manifest(cfg, dest)
    write_manifest(manifest, manifest_path)
    log.info("manifest_written", path=str(manifest_path), files=manifest.file_count)
    return FetchResult(manifest=manifest, cloned=cloned)
