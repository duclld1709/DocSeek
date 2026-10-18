"""Experiment runner (PROJECT_SPEC §5.8): ``docseek eval run`` and ``docseek eval compare``.

A run evaluates one retrieval configuration on one golden split and writes
``experiments/<ts>_<name>/``:

- ``metrics.json``: means with bootstrap 95% CIs, overall and per ``qtype``, plus Leakage,
  StaleHit and latency percentiles.
- ``per_query.jsonl``: hits, the rank of every relevant section and per-query metrics.
- ``traces.jsonl``: one ``retrieve`` event per query.
- ``env.json``: git sha, config and its hash, golden-set hash, seed, library versions.

Queries without relevant sections (unanswerable, or permission queries for a principal
without access) are excluded from ranking metrics but always count for Leakage.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from docseek.core.acl import AccessPolicy
from docseek.core.interfaces import Retriever
from docseek.core.models import Chunk, Section
from docseek.core.runs import environment_info, make_run_dir, write_json
from docseek.core.tracing import JsonlTraceSink, Tracer, trace_scope
from docseek.eval.golden import GoldenQuery, select
from docseek.eval.metrics import (
    RANKING_METRICS,
    hits_to_sections,
    leakage_at_k,
    stale_hit_at_k,
)
from docseek.eval.sanity import OracleRetriever, RandomRetriever
from docseek.eval.stats import DEFAULT_SEED, bootstrap_ci, compare_paired


class RetrieverConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    retriever: RetrieverConfig
    k: int = Field(default=10, ge=1)
    cutoffs: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    golden: Path = Path("data/golden/golden_v1.jsonl")
    processed_dir: Path = Path("data/processed")
    chunks_file: str = "chunks.jsonl"  # M5 chunking ablation: chunks_<strategy>.jsonl
    # Evaluate only these golden qtypes (e.g. decompose on multihop) to save LLM calls.
    qtypes: list[str] | None = None
    seed: int = DEFAULT_SEED
    bootstrap_resamples: int = Field(default=1000, ge=100)

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def load_experiment(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as fh:
        return ExperimentConfig.model_validate(yaml.safe_load(fh))


RetrieverFactory = Callable[
    [RetrieverConfig, list[Chunk], list[GoldenQuery], AccessPolicy, int], Retriever
]


def _random(
    cfg: RetrieverConfig,
    chunks: list[Chunk],
    _g: list[GoldenQuery],
    policy: AccessPolicy,
    seed: int,
) -> Retriever:
    return RandomRetriever(chunks, policy, seed=int(getattr(cfg, "seed", seed)))


def _oracle(
    _cfg: RetrieverConfig,
    chunks: list[Chunk],
    golden: list[GoldenQuery],
    policy: AccessPolicy,
    _seed: int,
) -> Retriever:
    return OracleRetriever({q.qid: q.relevant for q in golden}, chunks, policy)


def _lexical(kind: str) -> RetrieverFactory:
    def factory(
        cfg: RetrieverConfig,
        chunks: list[Chunk],
        _g: list[GoldenQuery],
        policy: AccessPolicy,
        _seed: int,
    ) -> Retriever:
        from docseek.lexical import factory as lex

        builder = {"bm25": lex.build_bm25, "tfidf": lex.build_tfidf, "bm25s": lex.build_bm25s}
        return builder[kind](cfg.model_dump(exclude={"type"}), chunks, policy)

    return factory


def _dense(
    cfg: RetrieverConfig,
    chunks: list[Chunk],
    _g: list[GoldenQuery],
    policy: AccessPolicy,
    _seed: int,
) -> Retriever:
    from docseek.dense.retriever import build_dense

    return build_dense(cfg.model_dump(exclude={"type"}), chunks, policy)


def build_child(
    raw: Any, chunks: list[Chunk], golden: list[GoldenQuery], policy: AccessPolicy, seed: int
) -> Retriever:
    cfg = RetrieverConfig.model_validate(raw)
    try:
        factory = RETRIEVERS[cfg.type]
    except KeyError:
        raise ValueError(f"unknown retriever type {cfg.type!r}") from None
    return factory(cfg, chunks, golden, policy, seed)


def _hybrid(
    cfg: RetrieverConfig,
    chunks: list[Chunk],
    golden: list[GoldenQuery],
    policy: AccessPolicy,
    seed: int,
) -> Retriever:
    from docseek.hybrid.fusion import HybridRetriever

    opts = cfg.model_dump(exclude={"type"})
    children = {
        label: build_child(raw, chunks, golden, policy, seed)
        for label, raw in dict(opts["children"]).items()
    }
    method = str(opts.get("method", "rrf"))
    weights = opts.get("weights")
    if opts.get("alpha") is not None:  # two children: alpha on the first, 1 - alpha on the second
        first, second = list(children)
        alpha = float(opts["alpha"])
        weights = {first: alpha, second: 1.0 - alpha}
    tag = f"rrf{opts.get('rrf_k', 60)}" if method == "rrf" else f"w{opts.get('norm', 'minmax')}"
    return HybridRetriever(
        f"hybrid-{tag}-" + "+".join(children),
        children,
        method=method,
        depth=int(opts.get("depth", 50)),
        rrf_k=float(opts.get("rrf_k", 60)),
        weights=weights,
        norm=str(opts.get("norm", "minmax")),
    )


def _rerank(
    cfg: RetrieverConfig,
    chunks: list[Chunk],
    golden: list[GoldenQuery],
    policy: AccessPolicy,
    seed: int,
) -> Retriever:
    from docseek.core.config import load_models_config
    from docseek.rerank.cross_encoder import (
        CrossEncoderReranker,
        RerankedRetriever,
        rerank_texts,
    )

    opts = cfg.model_dump(exclude={"type"})
    first = build_child(opts["first_stage"], chunks, golden, policy, seed)
    model = str(opts.get("model") or load_models_config().reranker.default)
    cache = opts.get("cache_path", "data/index/rerank_scores.sqlite")
    max_chars = opts.get("max_chars")
    reranker = CrossEncoderReranker(
        model,
        cache_path=Path(cache) if cache else None,
        max_chars=int(max_chars) if max_chars else None,
    )
    top_n = int(opts.get("top_n", 50))
    return RerankedRetriever(
        f"{first.name}+{reranker.name}@{top_n}",
        first,
        reranker,
        rerank_texts(chunks),
        top_n=top_n,
    )


def _qdrant_hybrid(
    cfg: RetrieverConfig,
    chunks: list[Chunk],
    _g: list[GoldenQuery],
    policy: AccessPolicy,
    _seed: int,
) -> Retriever:
    from docseek.core.config import Settings
    from docseek.dense.qdrant_store import connect
    from docseek.dense.retriever import corpus_digest, make_embedder
    from docseek.hybrid.qdrant_native import QdrantHybridRetriever, collection_name
    from docseek.lexical.tokenizer import TokenizerConfig

    opts = cfg.model_dump(exclude={"type"})
    model = str(opts["model"])
    cache = opts.get("cache_dir", "data/index/emb")
    emb = make_embedder(model, Path(cache) if cache else None)
    k1, b = float(opts.get("k1", 1.2)), float(opts.get("b", 0.75))
    tok = TokenizerConfig(**dict(opts.get("tokenizer") or {}))
    fusion = str(opts.get("fusion", "rrf"))
    rrf_k = opts.get("rrf_k")
    retriever = QdrantHybridRetriever(
        f"qdrant-{fusion}-{model.split('/')[-1]}",
        connect(str(opts.get("location") or Settings().qdrant_url)),
        collection_name("hybrid", emb.name, corpus_digest(chunks, emb.name), f"{k1}-{b}", str(tok)),
        emb,
        policy,
        tokenizer=tok,
        fusion=fusion,
        depth=int(opts.get("depth", 50)),
        rrf_k=int(rrf_k) if rrf_k else None,
    )
    retriever.build(chunks, k1=k1, b=b)
    return retriever


def _graph(
    cfg: RetrieverConfig,
    chunks: list[Chunk],
    _g: list[GoldenQuery],
    policy: AccessPolicy,
    _seed: int,
) -> Retriever:
    from docseek.graph.graph_retriever import build_graph_retriever

    return build_graph_retriever(cfg.model_dump(exclude={"type"}), chunks, policy)


def _clip(
    cfg: RetrieverConfig,
    chunks: list[Chunk],
    _g: list[GoldenQuery],
    policy: AccessPolicy,
    _seed: int,
) -> Retriever:
    from docseek.dense.image_retriever import build_image_retriever

    return build_image_retriever(cfg.model_dump(exclude={"type"}), chunks, policy)


def _pipeline(kind: str) -> RetrieverFactory:
    def factory(
        cfg: RetrieverConfig,
        chunks: list[Chunk],
        golden: list[GoldenQuery],
        policy: AccessPolicy,
        seed: int,
    ) -> Retriever:
        from docseek.eval.pipelines import BUILDERS

        return BUILDERS[kind](cfg.model_dump(exclude={"type"}), chunks, golden, policy, seed)

    return factory


RETRIEVERS: dict[str, RetrieverFactory] = {
    "random": _random,
    "oracle": _oracle,
    "bm25": _lexical("bm25"),
    "tfidf": _lexical("tfidf"),
    "bm25s": _lexical("bm25s"),
    "dense": _dense,
    "hybrid": _hybrid,
    "rerank": _rerank,
    "qdrant_hybrid": _qdrant_hybrid,
    "graph": _graph,
    "clip": _clip,
    **{kind: _pipeline(kind) for kind in ("freshness", "expand", "filtered", "router")},
}


def build_retriever(
    cfg: ExperimentConfig, chunks: list[Chunk], golden: list[GoldenQuery], policy: AccessPolicy
) -> Retriever:
    try:
        factory = RETRIEVERS[cfg.retriever.type]
    except KeyError:
        raise ValueError(
            f"unknown retriever type {cfg.retriever.type!r}; known: {sorted(RETRIEVERS)}"
        ) from None
    return factory(cfg.retriever, chunks, golden, policy, cfg.seed)


@dataclass
class QueryResult:
    qid: str
    qtype: str
    principal: str
    split: str
    ranked: list[str]
    hits: list[dict[str, Any]]
    relevant_ranks: dict[str, int | None]
    metrics: dict[str, float]
    latency_ms: float

    def to_json(self) -> dict[str, Any]:
        return self.__dict__


def evaluate_query(
    q: GoldenQuery,
    retriever: Retriever,
    *,
    k: int,
    cutoffs: Sequence[int],
    section_acl: Mapping[str, Sequence[str]],
    policy: AccessPolicy,
    tracer: Tracer,
) -> QueryResult:
    with trace_scope():
        start = time.perf_counter()
        hits = retriever.search(q.to_query(), k)
        latency_ms = (time.perf_counter() - start) * 1000
        tracer.record(
            kind="retrieve",
            name=retriever.name,
            input={"qid": q.qid, "text": q.text, "principal": q.principal, "k": k},
            output={"hits": [h.chunk_id for h in hits]},
            latency_ms=latency_ms,
        )
    ranked = hits_to_sections(hits)
    metrics: dict[str, float] = {
        f"leakage@{k}": float(leakage_at_k(ranked, section_acl, policy.groups(q.principal), k))
    }
    if q.relevant:
        for name, fn in RANKING_METRICS.items():
            for c in cutoffs:
                metrics[f"{name}@{c}"] = fn(ranked, q.relevant, c)
        if q.stale:
            for c in cutoffs:
                metrics[f"stale_hit@{c}"] = stale_hit_at_k(ranked, q.stale, q.relevant, c)
    position = {sid: i for i, sid in enumerate(ranked, start=1)}
    return QueryResult(
        qid=q.qid,
        qtype=str(q.qtype),
        principal=q.principal,
        split=q.split,
        ranked=ranked,
        hits=[
            {"chunk_id": h.chunk_id, "section_id": h.section_id, "rank": h.rank, "score": h.score}
            for h in hits
        ],
        relevant_ranks={sid: position.get(sid) for sid in q.relevant},
        metrics=metrics,
        latency_ms=latency_ms,
    )


def aggregate(results: Sequence[QueryResult], *, n_resamples: int, seed: int) -> dict[str, Any]:
    """Mean + bootstrap CI of every metric, overall and per qtype; Leakage as a total."""

    def summarize(rs: Sequence[QueryResult]) -> dict[str, Any]:
        values: defaultdict[str, list[float]] = defaultdict(list)
        for r in rs:
            for name, v in r.metrics.items():
                values[name].append(v)
        out: dict[str, Any] = {"n_queries": len(rs)}
        for name in sorted(values):
            if name.startswith("leakage@"):
                out[name] = {"total": int(sum(values[name])), "max": int(max(values[name]))}
                continue
            ci = bootstrap_ci(values[name], n_resamples=n_resamples, seed=seed)
            out[name] = {"mean": ci.mean, "ci_low": ci.low, "ci_high": ci.high, "n": ci.n}
        return out

    by_type: defaultdict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        by_type[r.qtype].append(r)
    latencies = np.asarray([r.latency_ms for r in results]) if results else np.zeros(1)
    return {
        "overall": summarize(results),
        "by_qtype": {t: summarize(rs) for t, rs in sorted(by_type.items())},
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(latencies.max()),
        },
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(
    cfg: ExperimentConfig,
    *,
    golden: list[GoldenQuery],
    sections: list[Section],
    chunks: list[Chunk],
    policy: AccessPolicy,
    split: Literal["dev", "test", "all"],
    experiments_dir: Path,
    golden_path: Path | None = None,
) -> Path:
    """Evaluate ``cfg`` on ``split`` and write the run directory; returns its path."""
    queries = select(golden, split=split)
    if cfg.qtypes is not None:
        queries = [q for q in queries if str(q.qtype) in cfg.qtypes]
    if not queries:
        raise ValueError(f"no golden queries in split {split!r}")
    run_dir = make_run_dir(experiments_dir, f"{cfg.name}_{split}")
    tracer = Tracer(JsonlTraceSink(run_dir / "traces.jsonl"))
    from docseek.eval.pipelines import RUN_TRACER

    token = RUN_TRACER.set(tracer)  # LLM-backed retrievers trace into this run
    try:
        retriever = build_retriever(cfg, chunks, golden, policy)
        section_acl = {s.section_id: s.acl for s in sections}
        results = [
            evaluate_query(
                q,
                retriever,
                k=cfg.k,
                cutoffs=cfg.cutoffs,
                section_acl=section_acl,
                policy=policy,
                tracer=tracer,
            )
            for q in queries
        ]
    finally:
        RUN_TRACER.reset(token)
    metrics = aggregate(results, n_resamples=cfg.bootstrap_resamples, seed=cfg.seed)
    metrics.update(name=cfg.name, retriever=retriever.name, split=split, k=cfg.k)
    write_json(run_dir / "metrics.json", metrics)
    with (run_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in results:
            fh.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
    write_json(
        run_dir / "env.json",
        environment_info(
            command="eval run",
            config=cfg.model_dump(mode="json"),
            config_hash=cfg.digest(),
            split=split,
            seed=cfg.seed,
            golden_sha256=_file_sha256(golden_path) if golden_path else None,
            n_sections=len(sections),
            n_chunks=len(chunks),
        ),
    )
    return run_dir


def load_per_query(run_dir: Path) -> dict[str, dict[str, Any]]:
    with (run_dir / "per_query.jsonl").open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {r["qid"]: r for r in rows}


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    n: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    verdict: str


def compare_runs(
    run_a: Path,
    run_b: Path,
    *,
    metrics: Sequence[str] | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[list[MetricComparison], list[tuple[str, str, float]]]:
    """Paired comparison on the queries both runs share.

    Returns per-metric comparisons and, for the first metric, every shared query's
    ``(qid, qtype, delta)`` sorted from biggest loss to biggest win of B over A.
    """
    a, b = load_per_query(run_a), load_per_query(run_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("the two runs share no queries")
    names = (
        list(metrics)
        if metrics
        else sorted(
            {m for qid in shared for m in a[qid]["metrics"] if not m.startswith("leakage@")}
        )
    )
    out: list[MetricComparison] = []
    for name in names:
        qids = [q for q in shared if name in a[q]["metrics"] and name in b[q]["metrics"]]
        if not qids:
            continue
        xa = [a[q]["metrics"][name] for q in qids]
        xb = [b[q]["metrics"][name] for q in qids]
        c = compare_paired(xa, xb, seed=seed)
        out.append(
            MetricComparison(
                name,
                len(qids),
                c.mean_a,
                c.mean_b,
                c.delta,
                c.delta_ci.low,
                c.delta_ci.high,
                c.p_value,
                c.verdict,
            )
        )
    first = names[0] if names else None
    swings = sorted(
        (
            (q, a[q]["qtype"], b[q]["metrics"][first] - a[q]["metrics"][first])
            for q in shared
            if first and first in a[q]["metrics"] and first in b[q]["metrics"]
        ),
        key=lambda t: t[2],
    )
    return out, swings
