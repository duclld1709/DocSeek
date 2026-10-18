"""Stable identifiers for documents, sections and chunks.

Format (see PROJECT_SPEC §3.5):

- ``doc_id     = "<source>:<relative_path>"``
- ``section_id = "<doc_id>#<anchor>"``  -- the unit relevance is judged at
- ``chunk_id   = "<section_id>::<strategy>::<n>"``  -- the unit that gets indexed
"""

from __future__ import annotations

import re

_SOURCE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_STRATEGY_RE = re.compile(r"^[a-z0-9][a-z0-9_+-]*$")
_CHUNK_SEP = "::"


def make_doc_id(source: str, relative_path: str) -> str:
    """Build a document id from a source name and a path relative to the source root.

    Backslashes are normalised to forward slashes so ids are identical on every OS.

    Raises:
        ValueError: if the source is malformed or the path is absolute or escapes the root.
    """
    if not _SOURCE_RE.match(source):
        raise ValueError(f"invalid source {source!r}: expected lowercase [a-z0-9_-]")
    path = relative_path.replace("\\", "/")
    parts = path.split("/")
    is_absolute = path.startswith("/") or ":" in parts[0]  # POSIX root or Windows drive
    if not path or is_absolute or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"invalid relative path {relative_path!r}")
    if "#" in path:
        raise ValueError(f"relative path must not contain '#': {relative_path!r}")
    return f"{source}:{path}"


def make_section_id(doc_id: str, anchor: str) -> str:
    """Build a section id from its document id and heading anchor."""
    if ":" not in doc_id or "#" in doc_id:
        raise ValueError(f"invalid doc_id {doc_id!r}")
    if not anchor or "#" in anchor or _CHUNK_SEP in anchor or any(c.isspace() for c in anchor):
        raise ValueError(f"invalid anchor {anchor!r}")
    return f"{doc_id}#{anchor}"


def make_chunk_id(section_id: str, strategy: str, n: int) -> str:
    """Build a chunk id from its section id, the chunking strategy and its ordinal."""
    if "#" not in section_id:
        raise ValueError(f"invalid section_id {section_id!r}")
    if not _STRATEGY_RE.match(strategy):
        raise ValueError(f"invalid strategy {strategy!r}")
    if n < 0:
        raise ValueError(f"chunk ordinal must be >= 0, got {n}")
    return f"{section_id}{_CHUNK_SEP}{strategy}{_CHUNK_SEP}{n}"


def section_id_of_chunk(chunk_id: str) -> str:
    """Recover the section id a chunk belongs to (used to map chunk hits to sections in eval)."""
    section_id, sep, _ = chunk_id.rpartition(_CHUNK_SEP)
    section_id, sep2, _ = section_id.rpartition(_CHUNK_SEP)
    if not sep or not sep2 or "#" not in section_id:
        raise ValueError(f"invalid chunk_id {chunk_id!r}")
    return section_id


def doc_id_of_section(section_id: str) -> str:
    """Recover the document id a section belongs to."""
    doc_id, sep, _ = section_id.rpartition("#")
    if not sep or ":" not in doc_id:
        raise ValueError(f"invalid section_id {section_id!r}")
    return doc_id
