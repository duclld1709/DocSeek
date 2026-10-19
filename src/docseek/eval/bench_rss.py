"""Process RSS of one retrieval config, measured in a fresh interpreter (used by `docseek bench`).

    python -m docseek.eval.bench_rss <experiment.yaml> <chunks_file> <golden.jsonl>

Prints one JSON line: RSS after imports, after building the retriever (production settings,
docker Qdrant), and after answering 20 test queries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    import psutil

    from docseek.core.acl import AccessPolicy
    from docseek.core.config import load_principals
    from docseek.core.models import Chunk, Query
    from docseek.eval.golden import load_golden, select
    from docseek.eval.runner import build_retriever, load_experiment
    from docseek.ingest.pipeline import read_jsonl

    proc = psutil.Process()

    def mb() -> float:
        return float(proc.memory_info().rss) / 2**20

    cfg_path, chunks_file, golden_path = sys.argv[1:4]
    golden = load_golden(Path(golden_path))
    chunks = read_jsonl(Path("data/processed") / chunks_file, Chunk)
    policy = AccessPolicy(load_principals())
    base = mb()
    cfg = load_experiment(Path(cfg_path))
    retriever = build_retriever(cfg, chunks, golden, policy)
    built = mb()
    for q in select(golden, split="test")[:20]:
        retriever.search(Query(qid=q.qid, text=q.text, principal=q.principal), 10)
    out = {"rss_mb_base": base, "rss_mb_built": built, "rss_mb_after_queries": mb()}
    sys.stdout.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
