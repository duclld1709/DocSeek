"""DocSeek HTTP API.

**Auth is simulated (demo only):** the principal comes from the ``X-Principal`` header
(default ``anonymous``); a real deployment would derive it from a verified token. A
``principal`` in the request body is accepted only if it equals the header, so a client
cannot raise its own access by editing the body. Unknown principals get 403. Traces are
private to the principal that created them.

    uv run uvicorn docseek.api.app:app --port 8000      # OpenAPI docs at /docs
"""

import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from docseek.agent.answer import AnswerResult
from docseek.agent.runtime import ANSWER_MODES, Runtime
from docseek.core.config import UnknownPrincipalError
from docseek.core.models import Query, TraceEvent
from docseek.core.tracing import JsonlTraceSink, Tracer, trace_scope
from docseek.llm.client import CachedLLMClient

SearchMode = Literal["auto", "bm25", "dense", "hybrid", "rerank"]
AskMode = Literal["one-shot", "agentic", "agentic-no-fs"]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    principal: str | None = Field(default=None, description="Must match X-Principal if set.")
    mode: SearchMode = "auto"
    k: int = Field(default=10, ge=1, le=50)
    debug: bool = False


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    principal: str | None = Field(default=None, description="Must match X-Principal if set.")
    mode: AskMode = "agentic"
    k: int = Field(default=5, ge=1, le=20)
    debug: bool = False


class SearchHit(BaseModel):
    rank: int
    section_id: str
    title: str
    snippet: str
    score: float
    modality: str = "text"
    page: int | None = None
    file: str | None = None
    retriever: str


class SearchResponse(BaseModel):
    trace_id: str
    principal: str
    mode: str
    hits: list[SearchHit]
    latency_ms: float


class AskResponse(AnswerResult):
    principal: str


class TraceStore:
    """Trace sink kept per trace id (bounded) and optionally appended to a JSONL file."""

    def __init__(self, max_traces: int = 500, path: Path | None = None) -> None:
        self._events: OrderedDict[str, list[TraceEvent]] = OrderedDict()
        self._owner: dict[str, str] = {}
        self._max = max_traces
        self._file = JsonlTraceSink(path) if path is not None else None
        self._lock = Lock()

    def write(self, event: TraceEvent) -> None:
        with self._lock:
            self._events.setdefault(event.trace_id, []).append(event)
            self._events.move_to_end(event.trace_id)
            while len(self._events) > self._max:
                old, _ = self._events.popitem(last=False)
                self._owner.pop(old, None)
        if self._file is not None:
            self._file.write(event)

    def own(self, trace_id: str, principal: str) -> None:
        with self._lock:
            self._owner[trace_id] = principal

    def get(self, trace_id: str, principal: str) -> list[TraceEvent] | None:
        with self._lock:
            if self._owner.get(trace_id) != principal:
                return None  # someone else's trace looks like a missing one
            return list(self._events.get(trace_id, []))


def create_app(
    runtime_factory: Callable[[], Runtime],
    llm_factory: Callable[[Tracer], CachedLLMClient],
    answer_model: str,
    *,
    trace_path: Path | None = None,
) -> FastAPI:
    app = FastAPI(
        title="DocSeek",
        version="1.0.0",
        description="Agentic retrieval over the FastAPI docs and an internal KB. "
        "Auth is simulated with the X-Principal header (demo only).",
    )
    store = TraceStore(path=trace_path)
    tracer = Tracer(store)
    state: dict[str, Any] = {}
    lock = Lock()

    def runtime() -> Runtime:
        with lock:
            if "rt" not in state:
                state["rt"] = runtime_factory()
                state["llm"] = llm_factory(tracer)
        rt: Runtime = state["rt"]
        return rt

    def principal_of(body_principal: str | None, header: str | None, rt: Runtime) -> str:
        principal = header or "anonymous"
        if body_principal is not None and body_principal != principal:
            raise HTTPException(403, "principal comes from the X-Principal header")
        try:
            rt.policy.groups(principal)
        except UnknownPrincipalError:
            raise HTTPException(403, f"unknown principal {principal!r}") from None
        return principal

    @app.get("/health")
    def health(rt: Annotated[Runtime, Depends(runtime)]) -> dict[str, Any]:
        return {
            "status": "ok",
            "chunks": len(rt.backend.chunks),
            "search_modes": [*rt.backend.retrievers, "rerank"],
            "answer_modes": list(ANSWER_MODES),
            "disabled": rt.disabled,
            "answer_model": answer_model,
        }

    @app.post("/search", response_model=SearchResponse)
    def search(
        req: SearchRequest,
        rt: Annotated[Runtime, Depends(runtime)],
        x_principal: Annotated[str | None, Header()] = None,
    ) -> SearchResponse:
        principal = principal_of(req.principal, x_principal, rt)
        retriever = (
            rt.one_shot_retriever if req.mode == "rerank" else rt.backend.retrievers.get(req.mode)
        )
        if retriever is None:
            raise HTTPException(400, f"mode {req.mode!r} is not configured")
        with trace_scope() as tid:
            store.own(tid, principal)
            start = time.perf_counter()
            raw = retriever.search(Query(qid="api", text=req.query, principal=principal), req.k)
            latency = (time.perf_counter() - start) * 1000
            hits: list[SearchHit] = []
            for h in raw:
                chunks = rt.backend.sections.get(h.section_id) or []
                chunk = next((c for c in chunks if c.chunk_id == h.chunk_id), None)
                if chunk is None or not rt.policy.can_read(principal, chunk.metadata["acl"]):
                    continue  # defence in depth; retrievers enforce ACL already
                meta = chunk.metadata
                hits.append(
                    SearchHit(
                        rank=len(hits) + 1,
                        section_id=h.section_id,
                        title=str(meta.get("section_path") or meta.get("title") or ""),
                        snippet=" ".join(chunk.text.split())[:400],
                        score=h.score,
                        modality=str(meta.get("modality", "text")),
                        page=meta.get("page"),
                        file=meta.get("file"),
                        retriever=h.retriever,
                    )
                )
            tracer.record(
                kind="retrieve",
                name=f"search.{req.mode}",
                input={"query": req.query, "principal": principal, "k": req.k},
                output={"hits": [h.section_id for h in hits]},
                latency_ms=latency,
            )
        return SearchResponse(
            trace_id=tid, principal=principal, mode=req.mode, hits=hits, latency_ms=latency
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(
        req: AskRequest,
        rt: Annotated[Runtime, Depends(runtime)],
        x_principal: Annotated[str | None, Header()] = None,
    ) -> AskResponse:
        principal = principal_of(req.principal, x_principal, rt)
        with trace_scope() as tid:
            store.own(tid, principal)
            result = rt.answer(req.mode, req.query, principal, state["llm"], answer_model)
            tracer.record(
                kind="answer",
                name=f"ask.{req.mode}",
                input={"query": req.query, "principal": principal},
                output={
                    "abstained": result.abstained,
                    "citations": result.citations,
                    "stop_reason": result.stop_reason,
                },
                latency_ms=result.latency_ms,
                cost_usd=0.0,
            )
        data = result.model_dump()
        if not req.debug:
            data["tool_calls"] = []
        data["trace_id"] = tid
        return AskResponse(**data, principal=principal)

    @app.get("/traces/{trace_id}")
    def traces(
        trace_id: str,
        rt: Annotated[Runtime, Depends(runtime)],
        x_principal: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        principal = principal_of(None, x_principal, rt)
        events = store.get(trace_id, principal)
        if not events:
            raise HTTPException(404, "trace not found")
        return {
            "trace_id": trace_id,
            "events": [e.model_dump(mode="json") for e in events],
            "total_latency_ms": sum(e.latency_ms for e in events if e.kind != "answer"),
            "total_cost_usd": sum(e.cost_usd for e in events),
        }

    return app


def _default_app() -> FastAPI:
    from docseek.agent.runtime import build_runtime
    from docseek.core.config import Settings, load_models_config
    from docseek.llm.client import build_llm_client

    return create_app(
        build_runtime,
        lambda tracer: build_llm_client(Settings(), tracer),
        load_models_config().llm.agent.model,
        trace_path=Path("data/traces/api.jsonl"),
    )


def __getattr__(name: str) -> Any:  # `uvicorn docseek.api.app:app` builds lazily
    if name == "app":
        return _default_app()
    raise AttributeError(name)
