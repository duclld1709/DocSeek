"""Run directories: every job that calls an LLM or evaluates writes ``experiments/<ts>_<name>/``.

``env.json`` records what is needed to reproduce or audit a run (PROJECT_SPEC §5.8): git sha
and dirty flag, library versions, machine, seed and any job-specific fields.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

_TRACKED_PACKAGES = (
    "docseek",
    "litellm",
    "numpy",
    "pydantic",
    "tokenizers",
    "ranx",
    "fastembed",
    "onnxruntime",
    "onnxruntime-directml",
    "qdrant-client",
    "hypothesis",
)


def make_run_dir(root: Path, name: str, *, now: datetime | None = None) -> Path:
    """Create ``<root>/<UTC timestamp>_<name>/`` (suffixing ``-1``, ``-2`` on collisions)."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    base = root / f"{stamp}_{name}"
    path, n = base, 0
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    path.mkdir(parents=True)
    return path


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, text=True, encoding="utf-8", check=False
        )
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def environment_info(**extra: Any) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    status = _git("status", "--porcelain")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": versions,
        # Which ONNX device fastembed models use (auto = DirectML GPU when installed).
        "onnx_providers_env": os.environ.get("DOCSEEK_ONNX_PROVIDERS", "auto"),
        **extra,
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
