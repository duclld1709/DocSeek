"""Resolve FastAPI's ``{* path hl[..] ln[..] title[..] *}`` code includes into code blocks.

Paths resolve relative to the mkdocs project directory (``docs/en/``), not the page. The
explicitly referenced file wins: authors sometimes include the non-``Annotated`` variant on
purpose, and ``ln[...]`` ranges only make sense for that exact file. Only when it does not
exist is the best available variant used, preferring ``_an_py310`` (PROJECT_SPEC §3.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

INCLUDE_RE = re.compile(r"^[ \t]*\{\*[ \t]+(?P<path>\S+)(?P<opts>.*?)[ \t]*\*\}[ \t]*$")
# Anywhere in text; whitespace after "{*" keeps Python's ``{**kwargs}`` from matching.
_INCLUDE_ANYWHERE_RE = re.compile(r"\{\*[ \t]+\S+[^\n]*?\*\}")
_OPT_RE = re.compile(r"(?P<key>hl|ln|title)\[(?P<value>[^\]]*)\]")
_VARIANT_RE = re.compile(r"^(?P<base>.+?)(?:_an)?(?:_py\d+)?\.py$")
VARIANT_PREFERENCE = ("_an_py310", "_py310", "_an_py39", "_py39", "_an", "")
_LANG_BY_SUFFIX = {".py": "python", ".toml": "toml", ".json": "json", ".txt": "text"}


@dataclass(frozen=True)
class ResolvedInclude:
    """Code block lines replacing one include directive."""

    lines: list[str]
    code_ref: str | None  # repo-relative path of the file actually used
    issue: str | None = None  # set when the include could not be resolved as written


@dataclass
class CodeIncludeResolver:
    """Resolves include directives against a checked-out repo.

    Args:
        repo_root: Root of the source repo; resolved files must stay inside it.
        mkdocs_dir: Directory include paths are relative to (``<repo>/docs/en``).
    """

    repo_root: Path
    mkdocs_dir: Path
    _cache: dict[Path, list[str]] = field(default_factory=dict, repr=False)

    def resolve(self, path: str, opts: str) -> ResolvedInclude:
        options = {m["key"]: m["value"] for m in _OPT_RE.finditer(opts)}
        root = self.repo_root.resolve()
        target = (self.mkdocs_dir / path).resolve()
        if not target.is_relative_to(root):
            return ResolvedInclude([f"(code include outside the repo: {path})"], None, "outside")
        issue = None
        if not target.is_file():
            fallback = best_variant(target)
            if fallback is None:
                ref = target.relative_to(root).as_posix()
                return ResolvedInclude(
                    [f"(code not available in this corpus: {ref})"], ref, "missing"
                )
            target, issue = fallback, "variant-fallback"
            options.pop("ln", None)  # ranges were written for the missing file
        ref = target.relative_to(root).as_posix()
        code = self._read(target)
        if "ln" in options:
            code = select_lines(code, options["ln"])
        lang = _LANG_BY_SUFFIX.get(target.suffix, target.suffix.lstrip("."))
        header = [f"File: {options['title'].strip(chr(34))}"] if "title" in options else []
        return ResolvedInclude([*header, f"```{lang}", *code, "```"], ref, issue)

    def _read(self, path: Path) -> list[str]:
        if path not in self._cache:
            self._cache[path] = path.read_text(encoding="utf-8").splitlines()
        return self._cache[path]


def find_unresolved_includes(text: str) -> list[str]:
    """Include directives still present in rendered text (must be empty after parsing)."""
    return _INCLUDE_ANYWHERE_RE.findall(text)


def best_variant(missing: Path) -> Path | None:
    """Existing sibling variant of ``missing`` in ``VARIANT_PREFERENCE`` order, if any."""
    m = _VARIANT_RE.match(missing.name)
    if not m or not missing.parent.is_dir():
        return None
    for suffix in VARIANT_PREFERENCE:
        candidate = missing.parent / f"{m['base']}{suffix}.py"
        if candidate.is_file():
            return candidate
    return None


def select_lines(lines: list[str], spec: str) -> list[str]:
    """Keep the 1-based inclusive ranges in ``spec`` (``"9:24"``, ``"2,5:8"``).

    Gaps between non-adjacent ranges are marked with ``# ...`` so the code does not read
    as if it were contiguous.
    """
    out: list[str] = []
    last = 0
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo_s, _, hi_s = part.partition(":")
        lo = max(1, int(lo_s))
        hi = min(len(lines), int(hi_s) if hi_s else lo)
        if lo > hi:
            continue
        if out and lo > last + 1:
            out.append("# ...")
        out.extend(lines[lo - 1 : hi])
        last = hi
    return out
