"""`docseek bench`: quality, latency, resources and vector-index tradeoffs (M8).

Per retrieval config:

- **quality**: a normal ``run_experiment`` on the chosen split (CIs, by qtype), one run dir;
- **cold build**: caches off (no BM25 pickle, no embedding cache, in-memory Qdrant, no
  reranker score cache) → wall time and process RSS growth;
- **latency**: the production retriever (docker Qdrant) with query-embedding and reranker
  caches off, one warm-up pass, then ``reps`` timed passes → p50/p95/p99.

Vector-index sweep (dense model, docker Qdrant, HNSW forced with ``indexing_threshold=0``,
otherwise Qdrant brute-forces collections this small and ``ef`` does nothing): float32 vs
scalar int8 (with and without rescoring) x ``hnsw_ef`` → p50/p95, ANN recall@10 against
exact search, nDCG@10 against the golden set, and vector RAM (computed: n x dim x bytes).
"""

from __future__ import annotations

import contextlib
import json
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from docseek.core.acl import AccessPolicy
from docseek.core.interfaces import Embedder
from docseek.core.models import Chunk, Query, Section
from docseek.eval.golden import GoldenQuery, select
from docseek.eval.metrics import ndcg_at_k
from docseek.eval.runner import ExperimentConfig, RetrieverConfig, build_retriever, run_experiment

Opts = dict[str, Any]


def percentiles(values_ms: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values_ms, dtype=float)
    if arr.size == 0:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "n": 0}
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "n": int(arr.size)}


def override(raw: Any, fn: Callable[[Opts], Opts]) -> Any:
    """Apply ``fn`` to every retriever block (a dict with ``type``) in a nested config."""
    if isinstance(raw, dict):
        out = {k: override(v, fn) for k, v in raw.items()}
        return fn(out) if "type" in out else out
    if isinstance(raw, list):
        return [override(v, fn) for v in raw]
    return raw


def cold(block: Opts) -> Opts:
    t = block["type"]
    if t in ("bm25", "tfidf", "bm25s"):
        return {**block, "index_dir": None}
    if t == "dense":
        return {**block, "cache_dir": None, "location": ":memory:"}
    if t == "clip":
        return {**block, "cache_dir": None, "location": ":memory:"}
    if t == "rerank":
        return {**block, "cache_path": None}
    return block


def uncached_queries(block: Opts) -> Opts:
    t = block["type"]
    if t in ("dense", "clip"):
        return {**block, "cache_dir": None}
    if t == "rerank":
        return {**block, "cache_path": None}
    return block


def rss_mb() -> float:
    import psutil

    return float(psutil.Process().memory_info().rss) / 2**20


def docker_stats(names: Sequence[str] = ("docseek-qdrant-1", "docseek-neo4j-1")) -> dict[str, Any]:
    try:
        out = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", *names],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    return {r["Name"]: {"mem": r["MemUsage"], "cpu": r["CPUPerc"]} for r in rows}


def docker_du(container: str, path: str) -> int | None:
    try:
        out = subprocess.run(
            ["docker", "exec", container, "du", "-sb", path],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def machine_info() -> dict[str, Any]:
    import psutil

    gpu = None
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip()
    return {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "cores_logical": psutil.cpu_count(),
        "cores_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": gpu,
        "python": platform.python_version(),
    }


def time_queries(
    search: Callable[[Query], Any], queries: Sequence[Query], reps: int
) -> list[float]:
    for q in queries:  # warm-up pass (model load, JIT, connection setup)
        search(q)
    lat: list[float] = []
    for _ in range(reps):
        for q in queries:
            t = time.perf_counter()
            search(q)
            lat.append((time.perf_counter() - t) * 1000)
    return lat


def bench_config(
    cfg: ExperimentConfig,
    *,
    golden: list[GoldenQuery],
    sections: list[Section],
    chunks: list[Chunk],
    policy: AccessPolicy,
    split: str,
    reps: int,
    experiments_dir: Path,
) -> dict[str, Any]:
    run = run_experiment(
        cfg,
        golden=golden,
        sections=sections,
        chunks=chunks,
        policy=policy,
        split=split,  # type: ignore[arg-type]
        experiments_dir=experiments_dir,
    )
    raw = cfg.retriever.model_dump()
    before = rss_mb()
    t = time.perf_counter()
    cold_cfg = cfg.model_copy(
        update={"retriever": RetrieverConfig.model_validate(override(raw, cold))}
    )
    build_retriever(cold_cfg, chunks, golden, policy)
    build_s = time.perf_counter() - t
    rss_growth = rss_mb() - before
    live_cfg = cfg.model_copy(
        update={"retriever": RetrieverConfig.model_validate(override(raw, uncached_queries))}
    )
    retriever = build_retriever(live_cfg, chunks, golden, policy)
    queries = [
        Query(qid=q.qid, text=q.text, principal=q.principal)
        for q in select(golden, split=split)  # type: ignore[arg-type]
    ]
    lat = time_queries(lambda q: retriever.search(q, cfg.k), queries, reps)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    return {
        "name": cfg.name,
        "run": run.name,
        "quality": {
            m: metrics["overall"].get(m) for m in ("ndcg@10", "recall@10", "mrr@10", "n_queries")
        },
        "by_qtype_ndcg@10": {
            t: v.get("ndcg@10", {}).get("mean") for t, v in metrics.get("by_qtype", {}).items()
        },
        "cold_build_s": build_s,
        "rss_growth_mb": rss_growth,
        "latency_ms": percentiles(lat),
        "reps": reps,
    }


def vector_sweep(
    chunks: Sequence[Chunk],
    queries: Sequence[GoldenQuery],
    embedder: Embedder,
    *,
    qdrant_url: str,
    efs: Sequence[int] = (16, 64, 256),
    reps: int = 3,
    k: int = 10,
) -> dict[str, Any]:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=qdrant_url, timeout=120)
    vecs = embedder.embed_documents([c.text_for_embedding for c in chunks])
    qvecs = np.stack([embedder.embed_query(q.text) for q in queries])
    n, dim = vecs.shape
    variants: dict[str, models.QuantizationConfig | None] = {
        "float32": None,
        "int8": models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, quantile=0.99, always_ram=True
            )
        ),
    }
    out: dict[str, Any] = {"n_vectors": int(n), "dim": int(dim), "variants": {}}
    exact_top: list[list[int]] | None = None
    for name, quant in variants.items():
        coll = f"bench_{name}"
        if client.collection_exists(coll):
            client.delete_collection(coll)
        t = time.perf_counter()
        client.create_collection(
            coll,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=1),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
            quantization_config=quant,
        )
        client.upload_collection(coll, vectors=vecs, ids=list(range(n)), wait=True)
        while client.get_collection(coll).status != models.CollectionStatus.GREEN:
            time.sleep(0.5)
        build_s = time.perf_counter() - t

        def search(qv: Any, params: models.SearchParams, c: str = coll) -> list[int]:
            res = client.query_points(c, query=qv.tolist(), limit=k, search_params=params)
            return [int(p.id) for p in res.points]

        if exact_top is None:
            exact_top = [search(qv, models.SearchParams(exact=True)) for qv in qvecs]
        modes = [("", None)] if quant is None else [("+rescore", True), ("-rescore", False)]
        for suffix, rescore in modes:
            for ef in efs:
                qparams = (
                    models.QuantizationSearchParams(rescore=rescore)
                    if rescore is not None
                    else None
                )
                params = models.SearchParams(hnsw_ef=ef, quantization=qparams)
                tops = [search(qv, params) for qv in qvecs]
                lat = time_queries(partial(search, params=params), list(qvecs), reps)
                recall = float(
                    np.mean(
                        [len(set(a) & set(b)) / k for a, b in zip(tops, exact_top, strict=True)]
                    )
                )
                ndcg = float(
                    np.mean(
                        [
                            ndcg_at_k(
                                list(dict.fromkeys(chunks[i].section_id for i in top)),
                                q.relevant,
                                k,
                            )
                            for top, q in zip(tops, queries, strict=True)
                        ]
                    )
                )
                out["variants"][f"{name}{suffix}@ef{ef}"] = {
                    "latency_ms": percentiles(lat),
                    "ann_recall@10": recall,
                    "ndcg@10": ndcg,
                }
        bytes_per = 4 if quant is None else 1
        out["variants"][f"{name}_index"] = {
            "build_s": build_s,
            "vector_ram_mb_computed": n * dim * bytes_per / 2**20,
            "disk_bytes": docker_du("docseek-qdrant-1", f"/qdrant/storage/collections/{coll}"),
        }
        client.delete_collection(coll)
    return out
