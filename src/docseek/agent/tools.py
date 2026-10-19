"""The agent's tools ("search skills"), bound to one principal chosen by the server.

Security model:

- The **principal is fixed when the toolbox is built** (from the API's ``X-Principal`` header
  or the eval's golden query). No tool takes a principal argument; a tool call that tries to
  pass one is refused, and every tool enforces ACL itself (retrievers, FS tools, graph).
- Every section a tool shows the model is recorded in ``evidence``. The answer step only keeps
  citations to sections in that set (citations must come from what was retrieved).
- Tool output is document text, i.e. untrusted data; it is returned inside
  ``<tool_output>`` delimiters and the system prompt says it is data, not instructions.
"""

from __future__ import annotations

import bisect
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docseek.core.acl import AccessPolicy
from docseek.core.interfaces import Retriever
from docseek.core.models import Chunk, Query
from docseek.core.tracing import Tracer

SNIPPET_CHARS = 400
SECTION_CHARS = 4000
SEARCH_MODES = ("auto", "bm25", "dense", "hybrid")
FS_TOOLS = ("ls", "find", "grep", "cat")
FORBIDDEN_ARGS = frozenset({"principal", "user", "acl", "groups", "role"})


class ToolError(Exception):
    """An error message safe to show to the model."""


@dataclass
class SearchBackend:
    """Shared, principal-independent resources (built once per process)."""

    retrievers: Mapping[str, Retriever]  # keys from SEARCH_MODES
    chunks: Sequence[Chunk]
    image: Retriever | None = None
    graph: Any | None = None  # GraphRetriever-like: search(Query, k) with TEMPLATES
    fs_root: Path | None = None
    sections: dict[str, list[Chunk]] = field(init=False)

    def __post_init__(self) -> None:
        self.sections = {}
        for c in sorted(self.chunks, key=lambda c: c.chunk_id):
            self.sections.setdefault(c.section_id, []).append(c)


def _spec(
    name: str, description: str, props: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "search": _spec(
        "search",
        "Search the documentation (FastAPI docs + internal engineering KB + scanned PDFs + "
        "screenshots). Start here. Returns the top sections with section_id, title and a "
        "snippet. mode: auto (default, best overall: hybrid lexical+dense with freshness), "
        "bm25 (exact identifiers, error codes, config keys), dense (paraphrased questions), "
        "hybrid. Rephrase and search again if the snippets do not answer the question.",
        {
            "query": {"type": "string", "description": "What to look for, in plain words."},
            "mode": {"type": "string", "enum": list(SEARCH_MODES)},
            "doc_type": {
                "type": "string",
                "description": "Optional filter, e.g. tutorial, reference, runbook, policy.",
            },
        },
        ["query"],
    ),
    "get_section": _spec(
        "get_section",
        "Read the full text of one section by its section_id (from search results). Use it "
        "when a snippet looks relevant but is cut off.",
        {"section_id": {"type": "string"}},
        ["section_id"],
    ),
    "graph_search": _spec(
        "graph_search",
        "Follow relations in the knowledge graph from an API name or concept (e.g. Depends, "
        "OAuth2PasswordBearer). pattern: shared_mentions (sections mentioning it), "
        "code_examples (code using it), kb_references (internal KB docs referencing it), "
        "linked_pages (pages linked from sections about it). Best for questions that combine "
        "two named things.",
        {
            "entity": {"type": "string"},
            "pattern": {
                "type": "string",
                "enum": ["shared_mentions", "code_examples", "kb_references", "linked_pages"],
            },
        },
        ["entity"],
    ),
    "image_search": _spec(
        "image_search",
        "Find screenshots/diagrams by what they show (e.g. 'Swagger UI authorize dialog'). "
        "Returns the sections that contain matching images.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    "ls": _spec(
        "ls",
        "List a directory of the docs tree (like a developer browsing the repo). '.' is the "
        "root: fastapi-docs/ and internal-kb/. Each directory has an _index.md overview.",
        {"path": {"type": "string"}},
        [],
    ),
    "find": _spec(
        "find",
        "Find files by glob pattern over the docs tree, e.g. '*security*' or 'internal-kb/*.md'.",
        {"pattern": {"type": "string"}},
        ["pattern"],
    ),
    "grep": _spec(
        "grep",
        "Search file contents for an exact string (case-insensitive), optionally under a "
        "path. Good for identifiers, error codes and config keys.",
        {"pattern": {"type": "string"}, "path": {"type": "string"}},
        ["pattern"],
    ),
    "cat": _spec(
        "cat",
        "Read a file of the docs tree, optionally only lines start_line..end_line.",
        {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        ["path"],
    ),
}


def wrap(name: str, payload: Any) -> str:
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f'<tool_output tool="{name}">\n{body}\n</tool_output>'


class ToolBox:
    """Tools for one request; ``principal`` is fixed here and never comes from the model."""

    def __init__(
        self,
        backend: SearchBackend,
        policy: AccessPolicy,
        principal: str,
        *,
        k: int = 5,
        use_fs: bool = True,
        tracer: Tracer | None = None,
    ) -> None:
        self._backend = backend
        self._policy = policy
        self.principal = principal
        self._groups = policy.groups(principal)  # unknown principal raises: fail closed
        self._k = k
        self._tracer = tracer
        self.evidence: dict[str, str] = {}  # section_id -> title, in first-seen order
        self._fs: Any | None = None
        self._toc: dict[str, list[tuple[int, str]]] = {}
        if use_fs and backend.fs_root is not None:
            from docseek.fs.tools import FsTools

            self._fs = FsTools(backend.fs_root, policy, principal, tracer=tracer)
            manifest = json.loads((backend.fs_root / "_manifest.json").read_text("utf-8"))
            self._toc = {
                f: [(s["line"], s["section_id"]) for s in meta["sections"]]
                for f, meta in manifest["files"].items()
            }
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "search": self._search,
            "get_section": self._get_section,
        }
        if backend.graph is not None:
            self._handlers["graph_search"] = self._graph_search
        if backend.image is not None:
            self._handlers["image_search"] = self._image_search
        if self._fs is not None:
            self._handlers |= {
                "ls": self._ls,
                "find": self._find,
                "grep": self._grep,
                "cat": self._cat,
            }

    @property
    def names(self) -> list[str]:
        return list(self._handlers)

    def specs(self) -> list[dict[str, Any]]:
        return [TOOL_SPECS[n] for n in self._handlers]

    def call(self, name: str, args: Mapping[str, Any]) -> str:
        """Run one tool call; errors come back as text for the model, never as exceptions."""
        start = time.perf_counter()
        try:
            if name not in self._handlers:
                raise ToolError(f"unknown tool {name!r}; available: {', '.join(self.names)}")
            bad = FORBIDDEN_ARGS & set(args)
            if bad:
                raise ToolError(
                    "the principal is set by the server and cannot be changed by a tool call "
                    f"(remove {sorted(bad)})"
                )
            out = wrap(name, self._handlers[name](dict(args)))
            error = None
        except ToolError as exc:
            out, error = wrap(name, {"error": str(exc)}), str(exc)
        except Exception as exc:  # FsError and friends: show the message, keep the loop alive
            out, error = wrap(name, {"error": f"{type(exc).__name__}: {exc}"}), str(exc)
        if self._tracer is not None:
            self._tracer.record(
                kind="tool_call",
                name=name,
                input={"args": dict(args), "principal": self.principal},
                output={"error": error, "chars": len(out), "evidence": len(self.evidence)},
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        return out

    # ---- helpers -------------------------------------------------------------------------
    def can_read(self, section_id: str) -> bool:
        chunks = self._backend.sections.get(section_id)
        if not chunks:
            return False
        return not self._groups.isdisjoint(chunks[0].metadata.get("acl") or [])

    def _note(self, section_id: str) -> None:
        chunks = self._backend.sections.get(section_id)
        if chunks and section_id not in self.evidence:
            self.evidence[section_id] = str(chunks[0].metadata.get("title", ""))

    def _hits(self, retriever: Retriever, query: Query) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for h in retriever.search(query, self._k * 3):
            if h.section_id in seen or not self.can_read(h.section_id):
                continue  # retrievers enforce ACL already; this is defence in depth
            seen.add(h.section_id)
            chunk = next(
                (c for c in self._backend.sections[h.section_id] if c.chunk_id == h.chunk_id),
                self._backend.sections[h.section_id][0],
            )
            meta = chunk.metadata
            hit: dict[str, Any] = {
                "section_id": h.section_id,
                "title": meta.get("section_path") or meta.get("title"),
                "snippet": " ".join(chunk.text.split())[:SNIPPET_CHARS],
            }
            for key in ("doc_type", "status", "updated_at", "page", "file", "image_path"):
                if meta.get(key) not in (None, ""):
                    hit[key] = meta[key]
            out.append(hit)
            self._note(h.section_id)
            if len(out) >= self._k:
                break
        return out

    # ---- tools ---------------------------------------------------------------------------
    def _search(self, args: dict[str, Any]) -> Any:
        text = str(args.get("query") or "").strip()
        if not text:
            raise ToolError("query is required")
        mode = str(args.get("mode") or "auto")
        if mode not in self._backend.retrievers:
            raise ToolError(f"mode must be one of {sorted(self._backend.retrievers)}")
        from docseek.core.models import MetadataFilter

        flt = MetadataFilter(doc_type=[str(args["doc_type"])]) if args.get("doc_type") else None
        query = Query(qid="agent", text=text, principal=self.principal, filters=flt)
        hits = self._hits(self._backend.retrievers[mode], query)
        if not hits and flt is not None:  # a wrong filter guess should not end the search
            hits = self._hits(
                self._backend.retrievers[mode], query.model_copy(update={"filters": None})
            )
        return hits or {"results": [], "hint": "no results; try other words or mode=bm25"}

    def _get_section(self, args: dict[str, Any]) -> Any:
        return self.read_section(str(args.get("section_id") or ""))

    def read_section(self, sid: str) -> dict[str, Any]:
        """Full text of a readable section (recorded as evidence); hidden = missing."""
        if not self.can_read(sid):  # hidden and missing look the same
            raise ToolError(f"no such section: {sid!r}")
        chunks = self._backend.sections[sid]
        meta = chunks[0].metadata
        self._note(sid)
        return {
            "section_id": sid,
            "title": meta.get("section_path") or meta.get("title"),
            "text": "\n\n".join(c.text for c in chunks)[:SECTION_CHARS],
            **{k: meta[k] for k in ("status", "updated_at", "page", "file") if meta.get(k)},
        }

    def _graph_search(self, args: dict[str, Any]) -> Any:
        entity = str(args.get("entity") or "").strip()
        if not entity:
            raise ToolError("entity is required")
        pattern = args.get("pattern")
        graph = self._backend.graph
        assert graph is not None
        query = Query(qid="agent", text=entity, principal=self.principal)
        if pattern:
            from docseek.graph.graph_retriever import TEMPLATES

            if pattern not in TEMPLATES:
                raise ToolError(f"pattern must be one of {sorted(TEMPLATES)}")
            hits = graph.search_templates(query, self._k, [str(pattern)])
        else:
            hits = graph.search(query, self._k)
        out = []
        for h in hits:
            if self.can_read(h.section_id):
                self._note(h.section_id)
                out.append(
                    {
                        "section_id": h.section_id,
                        "title": self.evidence[h.section_id],
                        "via": h.debug.get("via"),
                    }
                )
        return out or {"results": [], "hint": "entity not in the graph; use search"}

    def _image_search(self, args: dict[str, Any]) -> Any:
        assert self._backend.image is not None
        text = str(args.get("query") or "").strip()
        if not text:
            raise ToolError("query is required")
        return self._hits(
            self._backend.image, Query(qid="agent", text=text, principal=self.principal)
        )

    def _sections_in(self, rel: str, lines: Sequence[int]) -> None:
        toc = self._toc.get(rel)
        if not toc:
            return
        starts = [line for line, _ in toc]
        for n in lines:
            i = bisect.bisect_right(starts, n) - 1
            if i >= 0:
                self._note(toc[i][1])

    def _ls(self, args: dict[str, Any]) -> Any:
        assert self._fs is not None
        return self._fs.ls(str(args.get("path") or "."))

    def _find(self, args: dict[str, Any]) -> Any:
        assert self._fs is not None
        return self._fs.find(str(args.get("pattern") or "*"))

    def _grep(self, args: dict[str, Any]) -> Any:
        assert self._fs is not None
        hits = self._fs.grep(str(args.get("pattern") or ""), str(args.get("path") or "."))
        for h in hits:
            self._sections_in(h["path"], [h["line"]])
        return hits

    def _cat(self, args: dict[str, Any]) -> Any:
        assert self._fs is not None
        path = str(args.get("path") or "")
        start = args.get("start_line")
        end = args.get("end_line")
        text = self._fs.cat(path, int(start) if start else None, int(end) if end else None)
        rel = path.replace("\\", "/").strip().lstrip("./")
        first = int(start) if start else 1
        self._sections_in(rel, range(first, first + text.count("\n") + 1))
        return text
