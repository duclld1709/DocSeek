"""Turn mkdocs-flavoured markdown lines into plain index text and extract internal links.

Applied outside fenced code only (except :func:`strip_console_markup`):

- ``/// tip | Title`` and ``//// tab | Name`` openers become ``Tip: Title`` / ``Tab: Name``;
  closers (``///``) disappear.
- ``::: fastapi.X`` (mkdocstrings) becomes ``API reference: fastapi.X``; its indented
  ``options:`` block is dropped by the caller.
- ``<dfn title="T">X</dfn>`` and ``<abbr title="T">X</abbr>`` become ``X (T)`` so the
  definition stays searchable; ``<img>`` becomes ``[image: alt]``; other tags are removed.
- ``[text](target)`` keeps ``text``; ``![alt](src)`` becomes ``[image: alt]``.

Inline code spans are never rewritten.
"""

from __future__ import annotations

import html
import posixpath
import re
from collections.abc import Callable

ADMONITION_OPEN_RE = re.compile(
    r"^(?P<indent>[ \t]*)/{3,4}[ \t]*(?P<kind>[A-Za-z][\w-]*)"
    r"(?:[ \t]*\|[ \t]*(?P<title>.*?))?[ \t]*$"
)
ADMONITION_CLOSE_RE = re.compile(r"^[ \t]*/{3,4}[ \t]*$")
MKDOCSTRINGS_RE = re.compile(r"^:::[ \t]+(?P<name>\S+)[ \t]*$")

_CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1")
_TITLED_RE = re.compile(
    r"<(?P<tag>dfn|abbr)\b[^>]*?\btitle=\"(?P<title>[^\"]*)\"[^>]*>(?P<text>.*?)</(?P=tag)>",
    re.I | re.S,
)
_IMG_RE = re.compile(r"<img\b(?P<attrs>[^>]*)/?>", re.I)
_ALT_RE = re.compile(r"\balt=\"(?P<v>[^\"]*)\"", re.I)
_SRC_RE = re.compile(r"\bsrc=\"(?P<v>[^\"]*)\"", re.I)
_TAG_RE = re.compile(r"</?[A-Za-z][\w-]*(?:\s[^<>]*)?/?>")
_MD_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]*)[^)]*\)")
_MD_LINK_RE = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]*)(?:\s+\"[^\"]*\")?\)")
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.I)
_CONSOLE_INFO = {"console", "bash", "sh", "shell", "shell-session", "text", "txt"}


def admonition_line(line: str) -> str | None:
    """Rewritten opener/closer line, or ``None`` if ``line`` is not an admonition marker."""
    if ADMONITION_CLOSE_RE.match(line):
        return ""
    m = ADMONITION_OPEN_RE.match(line)
    if not m:
        return None
    label = m["kind"].replace("-", " ").capitalize()
    title = (m["title"] or "").strip()
    return f"{m['indent']}{label}: {title}" if title else f"{m['indent']}{label}:"


def normalize_inline(line: str) -> str:
    """Rewrite markdown links/images, then inline HTML, never touching code spans.

    Links are handled on the whole line first because link text may itself contain code
    (``[`orjson`](https://...)``); a link that starts inside a code span is left alone.
    """
    line = _sub_outside_code(
        _MD_IMAGE_RE, lambda m: f"[image: {m['alt'] or _basename(m['src'])}]", line
    )
    line = _sub_outside_code(_MD_LINK_RE, lambda m: m["text"], line)
    parts: list[str] = []
    pos = 0
    for m in _CODE_SPAN_RE.finditer(line):
        parts.append(_normalize_html(line[pos : m.start()]))
        parts.append(m.group(0))
        pos = m.end()
    parts.append(_normalize_html(line[pos:]))
    return "".join(parts)


def _normalize_html(text: str) -> str:
    if not text:
        return text
    text = _TITLED_RE.sub(lambda m: f"{_TAG_RE.sub('', m['text'])} ({m['title']})", text)
    text = _IMG_RE.sub(lambda m: _image_marker(m["attrs"]), text)
    text = _TAG_RE.sub("", text)
    return html.unescape(text)


def _code_spans(line: str) -> list[tuple[int, int]]:
    return [m.span() for m in _CODE_SPAN_RE.finditer(line)]


def _in_code(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(lo <= pos < hi for lo, hi in spans)


def _sub_outside_code(
    pattern: re.Pattern[str], repl: Callable[[re.Match[str]], str], line: str
) -> str:
    spans = _code_spans(line)
    return pattern.sub(lambda m: m.group(0) if _in_code(m.start(), spans) else repl(m), line)


def _image_marker(attrs: str) -> str:
    alt = _ALT_RE.search(attrs)
    if alt and alt["v"].strip():
        return f"[image: {alt['v'].strip()}]"
    src = _SRC_RE.search(attrs)
    return f"[image: {_basename(src['v']) if src else 'unnamed'}]"


def _basename(src: str) -> str:
    return src.rstrip("/").rsplit("/", 1)[-1] or src


def mkdocstrings_line(line: str) -> str | None:
    """``API reference: <name>`` for a ``::: <name>`` directive, else ``None``."""
    m = MKDOCSTRINGS_RE.match(line)
    return f"API reference: {m['name']}" if m else None


def is_console_fence(info: str) -> bool:
    """Whether a fence's info string marks shell output (where termy HTML appears)."""
    lang = info.strip().lstrip("{").strip().lstrip(".").split()[0] if info.strip() else ""
    return lang.lower() in _CONSOLE_INFO


def strip_console_markup(line: str) -> str:
    """Remove colouring tags like ``<font color=...>`` from shell transcripts."""
    return html.unescape(_TAG_RE.sub("", line))


def extract_links(line: str, *, rel_path: str, source: str) -> list[str]:
    """Internal link targets in ``line`` as ``doc_id`` or ``section_id`` strings.

    Relative ``*.md`` paths resolve against the page's folder; ``#anchor`` alone points
    into the same page. External URLs and paths escaping the docs root are ignored.
    """
    targets: list[str] = []
    spans = _code_spans(line)
    for m in _MD_LINK_RE.finditer(line):
        if _in_code(m.start(), spans) or (m.start() > 0 and line[m.start() - 1] == "!"):
            continue
        target = _resolve_link(m["target"], rel_path=rel_path, source=source)
        if target and target not in targets:
            targets.append(target)
    return targets


def _resolve_link(target: str, *, rel_path: str, source: str) -> str | None:
    if not target or _SCHEME_RE.match(target) or target.startswith("//"):
        return None
    path, _, anchor = target.partition("#")
    if not path:
        return f"{source}:{rel_path}#{anchor}" if anchor else None
    if not path.endswith(".md"):
        return None
    if path.startswith("/"):
        resolved = posixpath.normpath(path.lstrip("/"))
    else:
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(rel_path), path))
    if resolved.startswith("..") or resolved.startswith("/"):
        return None
    doc_id = f"{source}:{resolved}"
    return f"{doc_id}#{anchor}" if anchor else doc_id
