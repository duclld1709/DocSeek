"""Sandboxed, ACL-aware filesystem tools over ``data/projected`` (for the M8 agent).

- **Sandbox**: paths are relative to the projection root; absolute paths, drive letters,
  ``..`` components and symlinks anywhere on the path are rejected.
- **ACL**: a file the principal may not read behaves exactly like a file that does not exist
  (same error), and is skipped by ``ls``/``find``/``grep``. ``_index.md`` is rendered per
  principal from the manifest; the manifest itself is never served.
- **Limits**: output is capped (lines, characters, results); every call is traced.
- File contents are untrusted data: callers must wrap tool output in delimiters before it
  reaches an LLM (prompt-injection risk from document text).
"""

from __future__ import annotations

import fnmatch
import json
import re
import time
from pathlib import Path, PurePosixPath
from typing import Any

from docseek.core.acl import AccessPolicy
from docseek.core.tracing import Tracer
from docseek.ingest.projection import INDEX, MANIFEST, render_index

MAX_LINES = 200
MAX_CHARS = 8000
MAX_RESULTS = 50
MAX_PATTERN = 200


def _parent(rel: str) -> str:
    parent = PurePosixPath(rel).parent.as_posix()
    return "" if parent == "." else parent


class FsError(Exception):
    """A tool error safe to show to the agent (never reveals whether a hidden file exists)."""


class FsTools:
    def __init__(
        self, root: Path, policy: AccessPolicy, principal: str, *, tracer: Tracer | None = None
    ) -> None:
        self._root = root.resolve()
        self._groups = policy.groups(principal)  # unknown principal raises: fail closed
        self._tracer = tracer
        manifest = json.loads((self._root / MANIFEST).read_text(encoding="utf-8"))
        self._files: dict[str, Any] = {
            rel: meta
            for rel, meta in manifest["files"].items()
            if not self._groups.isdisjoint(meta["acl"])
        }

    # ---- path handling -------------------------------------------------------------------
    def _rel(self, path: str) -> str:
        p = path.replace("\\", "/").strip()
        if p in ("", ".", "/"):
            return ""
        pure = PurePosixPath(p)
        if pure.is_absolute() or re.match(r"^[A-Za-z]:", p) or ".." in pure.parts:
            raise FsError(f"path not allowed: {path!r}")
        rel = pure.as_posix().strip("/")
        current = self._root
        for part in PurePosixPath(rel).parts:
            current = current / part
            if current.is_symlink():
                raise FsError(f"path not allowed: {path!r}")
        resolved = (self._root / rel).resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise FsError(f"path not allowed: {path!r}")
        return rel

    def _visible_dirs(self) -> set[str]:
        dirs = {""}
        for rel in self._files:
            parts = PurePosixPath(rel).parts[:-1]
            for i in range(1, len(parts) + 1):
                dirs.add("/".join(parts[:i]))
        return dirs

    def _trace(self, op: str, args: dict[str, Any], out: Any, start: float) -> None:
        if self._tracer is not None:
            size = len(out) if isinstance(out, (list, str)) else 0
            self._tracer.record(
                kind="tool_call",
                name=f"fs.{op}",
                input=args,
                output={"size": size},
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    # ---- tools ---------------------------------------------------------------------------
    def ls(self, path: str = ".") -> list[str]:
        start = time.perf_counter()
        rel = self._rel(path)
        dirs = self._visible_dirs()
        if rel not in dirs:
            raise FsError(f"no such directory: {path!r}")
        prefix = f"{rel}/" if rel else ""
        entries: set[str] = set()
        for d in dirs:
            if d.startswith(prefix) and d != rel and "/" not in d[len(prefix) :]:
                entries.add(d[len(prefix) :] + "/")
        for f in self._files:
            if f.startswith(prefix) and "/" not in f[len(prefix) :]:
                entries.add(f[len(prefix) :])
        if rel:
            entries.add(INDEX)
        out = sorted(entries)
        self._trace("ls", {"path": path}, out, start)
        return out

    def find(self, pattern: str) -> list[str]:
        start = time.perf_counter()
        if len(pattern) > MAX_PATTERN:
            raise FsError("pattern too long")
        out = sorted(f for f in self._files if fnmatch.fnmatch(f, pattern))[:MAX_RESULTS]
        self._trace("find", {"pattern": pattern}, out, start)
        return out

    def grep(
        self, pattern: str, path: str = ".", *, regex: bool = False, max_results: int = MAX_RESULTS
    ) -> list[dict[str, Any]]:
        start = time.perf_counter()
        if len(pattern) > MAX_PATTERN:
            raise FsError("pattern too long")
        rel = self._rel(path)
        try:
            rx = re.compile(pattern if regex else re.escape(pattern), re.IGNORECASE)
        except re.error as exc:
            raise FsError(f"bad regex: {exc}") from None
        targets = [
            f for f in sorted(self._files) if f == rel or f.startswith(f"{rel}/" if rel else "")
        ]
        if rel and rel not in self._files and rel not in self._visible_dirs():
            raise FsError(f"no such file or directory: {path!r}")
        out: list[dict[str, Any]] = []
        limit = max(1, min(max_results, MAX_RESULTS))
        for f in targets:
            for i, line in enumerate((self._root / f).read_text(encoding="utf-8").splitlines(), 1):
                if rx.search(line[:2000]):
                    out.append({"path": f, "line": i, "text": line[:300]})
                    if len(out) >= limit:
                        self._trace("grep", {"pattern": pattern, "path": path}, out, start)
                        return out
        self._trace("grep", {"pattern": pattern, "path": path}, out, start)
        return out

    def cat(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        *,
        section_id: str | None = None,
    ) -> str:
        start = time.perf_counter()
        rel = self._rel(path)
        if PurePosixPath(rel).name == INDEX:
            d = _parent(rel)
            rels = [f for f in self._files if _parent(f) == d]
            if not rels:
                raise FsError(f"no such file: {path!r}")
            out = render_index(d, rels, self._files)[:MAX_CHARS]
            self._trace("cat", {"path": path}, out, start)
            return out
        if rel not in self._files:
            raise FsError(f"no such file: {path!r}")
        lines = (self._root / rel).read_text(encoding="utf-8").splitlines()
        if section_id is not None:
            toc = self._files[rel]["sections"]
            idx = next((i for i, t in enumerate(toc) if t["section_id"] == section_id), None)
            if idx is None:
                raise FsError(f"no section {section_id!r} in {path!r}")
            start_line = toc[idx]["line"]
            end_line = toc[idx + 1]["line"] - 1 if idx + 1 < len(toc) else len(lines)
        lo = max(1, start_line or 1)
        hi = min(len(lines), end_line or len(lines), lo + MAX_LINES - 1)
        body = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(lo, hi + 1))
        if len(body) > MAX_CHARS:
            body = body[:MAX_CHARS] + "\n[... truncated]"
        if hi < len(lines) and not section_id:
            body += f"\n[... {len(lines) - hi} more lines; use start_line={hi + 1}]"
        self._trace("cat", {"path": path, "start_line": lo, "end_line": hi}, body, start)
        return body
