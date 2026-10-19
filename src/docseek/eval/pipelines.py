"""Experiment-config builders for the M5 query-understanding wrappers.

Each wrapper takes an ``inner`` retriever config (any registered type) and adds one
component, so ablations are config edits. LLM-backed wrappers use the generator role from
``configs/models.yaml`` through one shared ``CachedLLMClient`` whose tracer is the current
run's (``RUN_TRACER``), so their cost lands in that run's ``traces.jsonl``.

Types:

- ``freshness``: {inner, beta, tau_days, deprecated_penalty, superseded_penalty, depth}
- ``expand``: {inner, mode: rewrite|hyde|decompose, n, depth}
- ``filtered``: {inner} — LLM-extracted MetadataFilter with empty-result fallback
- ``router``: {routes: {qtype: cfg}, default: cfg, use_llm}
- ``subset``: {modality: [..], inner} — inner built on the chunks of those modalities only
  (M7: the router's image route searches image chunks only)
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import cache
from typing import Any

from docseek.core.acl import AccessPolicy
from docseek.core.interfaces import Retriever
from docseek.core.models import Chunk, Query, QueryType, ScoredHit
from docseek.core.tracing import Tracer
from docseek.eval.golden import GoldenQuery
from docseek.llm.client import CachedLLMClient

RUN_TRACER: ContextVar[Tracer | None] = ContextVar("RUN_TRACER", default=None)


@cache
def _client(tracer_id: int) -> CachedLLMClient:  # one client per run tracer
    from docseek.core.config import Settings
    from docseek.llm.client import build_llm_client

    return build_llm_client(Settings(), RUN_TRACER.get())


def llm() -> CachedLLMClient:
    return _client(id(RUN_TRACER.get()))


def generator_role() -> Any:
    from docseek.core.config import load_models_config

    return load_models_config().llm.query_role


def _inner(
    opts: dict[str, Any],
    chunks: list[Chunk],
    golden: list[GoldenQuery],
    policy: AccessPolicy,
    seed: int,
    key: str = "inner",
) -> Retriever:
    from docseek.eval.runner import build_child

    return build_child(opts[key], chunks, golden, policy, seed)


def build_freshness(
    opts: dict[str, Any],
    chunks: list[Chunk],
    golden: list[GoldenQuery],
    policy: AccessPolicy,
    seed: int,
) -> Retriever:
    from docseek.query.freshness import FreshnessConfig, FreshnessReranker

    inner = _inner(opts, chunks, golden, policy, seed)
    cfg = FreshnessConfig(
        beta=float(opts.get("beta", 0.3)),
        tau_days=float(opts.get("tau_days", 180)),
        deprecated_penalty=float(opts.get("deprecated_penalty", 0.5)),
        superseded_penalty=float(opts.get("superseded_penalty", 0.5)),
    )
    return FreshnessReranker(
        f"{inner.name}+fresh",
        inner,
        chunks,
        cfg,
        depth=int(opts.get("depth", 50)),
        score_transform=str(opts.get("score_transform", "identity")),
    )


def build_expand(
    opts: dict[str, Any],
    chunks: list[Chunk],
    golden: list[GoldenQuery],
    policy: AccessPolicy,
    seed: int,
) -> Retriever:
    from docseek.query.rewrite import DecomposeRetriever, HyDERetriever, MultiQueryRetriever

    inner = _inner(opts, chunks, golden, policy, seed)
    mode = str(opts.get("mode", "rewrite"))
    depth = int(opts.get("depth", 50))
    name = f"{inner.name}+{mode}"
    if mode == "rewrite":
        return MultiQueryRetriever(
            name, inner, llm(), generator_role(), n=int(opts.get("n", 3)), depth=depth
        )
    if mode == "hyde":
        return HyDERetriever(name, inner, llm(), generator_role(), depth=depth)
    if mode == "decompose":
        return DecomposeRetriever(name, inner, llm(), generator_role(), depth=depth)
    raise ValueError(f"unknown expand mode {mode!r}")


class FilteredRetriever:
    """Extract a MetadataFilter from the query text (LLM) and search with fallback."""

    def __init__(self, name: str, inner: Retriever, doc_types: list[str]) -> None:
        from docseek.query.filters import FilterExtractor

        self.name = name
        self._inner = inner
        self._extractor = FilterExtractor(llm(), generator_role(), doc_types=doc_types)
        self.fallbacks = 0

    def search(self, query: Query, k: int) -> list[ScoredHit]:
        from docseek.query.filters import search_with_fallback

        flt = query.filters or self._extractor.extract(query.text)
        hits, fell_back = search_with_fallback(self._inner, query, flt, k, tracer=RUN_TRACER.get())
        self.fallbacks += fell_back
        return hits


def build_filtered(
    opts: dict[str, Any],
    chunks: list[Chunk],
    golden: list[GoldenQuery],
    policy: AccessPolicy,
    seed: int,
) -> Retriever:
    inner = _inner(opts, chunks, golden, policy, seed)
    doc_types = sorted({str(c.metadata.get("doc_type")) for c in chunks})
    return FilteredRetriever(f"{inner.name}+filters", inner, doc_types)


def build_router(
    opts: dict[str, Any],
    chunks: list[Chunk],
    golden: list[GoldenQuery],
    policy: AccessPolicy,
    seed: int,
) -> Retriever:
    from docseek.eval.runner import build_child
    from docseek.query.classifier import QueryClassifier
    from docseek.query.router import RouterRetriever

    default = build_child(opts["default"], chunks, golden, policy, seed)
    built: dict[str, Retriever] = {}
    routes: dict[QueryType, Retriever] = {}
    for qtype, raw in dict(opts.get("routes") or {}).items():
        key = repr(raw)  # identical route configs share one retriever
        if key not in built:
            built[key] = build_child(raw, chunks, golden, policy, seed)
        routes[QueryType(qtype)] = built[key]
    use_llm = bool(opts.get("use_llm", True))
    clf = QueryClassifier(llm() if use_llm else None, generator_role() if use_llm else None)
    return RouterRetriever(
        f"router({len(routes)} routes)",
        clf,
        routes,
        default,
        use_llm=use_llm,
        tracer=RUN_TRACER.get(),
    )


def build_subset(
    opts: dict[str, Any],
    chunks: list[Chunk],
    golden: list[GoldenQuery],
    policy: AccessPolicy,
    seed: int,
) -> Retriever:
    keep = set(opts["modality"])
    subset = [c for c in chunks if c.metadata.get("modality", "text") in keep]
    if not subset:
        raise ValueError(f"subset: no chunks with modality in {sorted(keep)}")
    return _inner(opts, subset, golden, policy, seed)


BUILDERS = {
    "freshness": build_freshness,
    "expand": build_expand,
    "filtered": build_filtered,
    "router": build_router,
    "subset": build_subset,
}
