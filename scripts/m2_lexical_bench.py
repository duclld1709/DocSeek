"""Own BM25 vs bm25s with the same tokenizer and k1/b: build time, index size, query latency
and top-10 overlap on the dev queries. Writes experiments/<ts>_m2_lexical_bench/bench.json.

    uv run python scripts/m2_lexical_bench.py
"""

from __future__ import annotations

import pickle
import statistics
import tempfile
import time
from pathlib import Path

from docseek.core.acl import AccessPolicy
from docseek.core.config import load_principals
from docseek.core.models import Chunk
from docseek.core.runs import environment_info, make_run_dir, write_json
from docseek.eval.golden import load_golden, select
from docseek.ingest.pipeline import read_jsonl
from docseek.lexical import factory


def _timed_build(build, n: int = 3):  # type: ignore[no-untyped-def]
    times, r = [], None
    for _ in range(n):
        t = time.perf_counter()
        r = build()
        times.append(time.perf_counter() - t)
    return r, min(times)


def main() -> None:
    chunks = read_jsonl(Path("data/processed/chunks.jsonl"), Chunk)
    policy = AccessPolicy(load_principals())
    queries = [
        q.to_query() for q in select(load_golden(Path("data/golden/golden_v1.jsonl")), split="dev")
    ]
    opts = {"k1": 1.5, "b": 0.5, "index_dir": None}
    own, own_build = _timed_build(lambda: factory.build_bm25(opts, chunks, policy))
    ref, ref_build = _timed_build(lambda: factory.build_bm25s(opts, chunks, policy))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "own.pkl"
        p.write_bytes(pickle.dumps(own._scorer.index))
        own_size = p.stat().st_size
        ref._scorer._model.save(Path(tmp) / "bm25s")
        ref_size = sum(f.stat().st_size for f in (Path(tmp) / "bm25s").rglob("*") if f.is_file())

    def lat(r):  # type: ignore[no-untyped-def]
        out = []
        for q in queries:
            t = time.perf_counter()
            r.search(q, 10)
            out.append((time.perf_counter() - t) * 1000)
        return out

    for r in (own, ref):  # warm-up
        lat(r)
    own_ms, ref_ms = lat(own), lat(ref)
    overlaps, identical = [], 0
    for q in queries:
        a = [h.chunk_id for h in own.search(q, 10)]
        b = [h.chunk_id for h in ref.search(q, 10)]
        overlaps.append(len(set(a) & set(b)) / max(len(a), len(b), 1))
        identical += a == b
    result = {
        "n_chunks": len(chunks),
        "n_queries": len(queries),
        "build_s": {"own": own_build, "bm25s": ref_build},
        "index_bytes": {"own_pickle": own_size, "bm25s_saved": ref_size},
        "query_ms_p50": {"own": statistics.median(own_ms), "bm25s": statistics.median(ref_ms)},
        "query_ms_p95": {
            "own": statistics.quantiles(own_ms, n=20)[-1],
            "bm25s": statistics.quantiles(ref_ms, n=20)[-1],
        },
        "top10_overlap_mean": statistics.mean(overlaps),
        "top10_identical_order": identical,
    }
    out = make_run_dir(Path("experiments"), "m2_lexical_bench")
    write_json(out / "bench.json", result)
    write_json(
        out / "env.json", environment_info(command="scripts/m2_lexical_bench.py", config=opts)
    )
    print(result)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
