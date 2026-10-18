"""Cold rerank latency per N on a fixed sample of dev queries (score cache disabled).

Quality comes from the cached eval runs; this measures what a user would wait for.
Writes experiments/<ts>_m4_rerank_latency/latency.json.

    uv run python scripts/m4_rerank_latency.py --n 20 50 100 --sample 20
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

from docseek.core.acl import AccessPolicy
from docseek.core.config import load_principals
from docseek.core.models import Chunk
from docseek.core.runs import environment_info, make_run_dir, write_json
from docseek.eval.golden import load_golden, select
from docseek.eval.runner import RetrieverConfig, build_child, load_experiment
from docseek.ingest.pipeline import read_jsonl
from docseek.rerank.cross_encoder import CrossEncoderReranker, rerank_texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/experiments/rerank_bge.yaml"))
    ap.add_argument("--n", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    opts = cfg.retriever.model_dump(exclude={"type"})
    chunks = read_jsonl(Path("data/processed/chunks.jsonl"), Chunk)
    policy = AccessPolicy(load_principals())
    golden = load_golden(cfg.golden)
    queries = [q.to_query() for q in select(golden, split="dev")]
    queries = random.Random(args.seed).sample(queries, min(args.sample, len(queries)))
    first = build_child(
        RetrieverConfig.model_validate(opts["first_stage"]), chunks, golden, policy, 0
    )
    reranker = CrossEncoderReranker(str(opts["model"]))  # no cache: cold scores
    texts = rerank_texts(chunks)
    reranker.score("warm-up", ["warm-up"])
    out: dict[str, dict[str, float]] = {}
    for q in queries:  # warm first stage (query-embedding cache)
        first.search(q, 10)
    t_first = []
    for q in queries:
        t = time.perf_counter()
        first.search(q, max(args.n))
        t_first.append((time.perf_counter() - t) * 1000)
    out["first_stage"] = {
        "p50": statistics.median(t_first),
        "p95": statistics.quantiles(t_first, n=20)[-1],
    }
    for n in args.n:
        times = []
        for q in queries:
            t = time.perf_counter()
            hits = first.search(q, n)
            reranker.rerank(q.text, hits[:n], texts, 10)
            times.append((time.perf_counter() - t) * 1000)
        out[f"rerank@{n}"] = {
            "p50": statistics.median(times),
            "p95": statistics.quantiles(times, n=20)[-1],
            "mean": statistics.mean(times),
        }
        print(n, out[f"rerank@{n}"], flush=True)
    run = make_run_dir(Path("experiments"), "m4_rerank_latency")
    write_json(run / "latency.json", {"n_queries": len(queries), "seed": args.seed, **out})
    write_json(run / "env.json", environment_info(command="scripts/m4_rerank_latency.py", n=args.n))
    print(f"-> {run}")


if __name__ == "__main__":
    main()
