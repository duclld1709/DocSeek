"""Cold-embed the corpus with each candidate model and index it in Qdrant.

Records dim, embed time (cache misses only), cache size and upsert time to
experiments/<ts>_m3_embed_bench/bench.json. Fills data/index/emb so eval runs reuse vectors.

    uv run python scripts/m3_embed_bench.py BAAI/bge-small-en-v1.5 nomic-ai/nomic-embed-text-v1.5
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from docseek.core.config import Settings
from docseek.core.models import Chunk
from docseek.core.runs import environment_info, make_run_dir, write_json
from docseek.dense.qdrant_store import QdrantStore, connect
from docseek.dense.retriever import corpus_digest, make_embedder
from docseek.ingest.pipeline import read_jsonl


def main(models: list[str]) -> None:
    chunks = read_jsonl(Path("data/processed/chunks.jsonl"), Chunk)
    texts = [c.text_for_embedding for c in chunks]
    client = connect(Settings().qdrant_url)
    out: dict[str, dict[str, float | int | str]] = {}
    for model in models:
        emb = make_embedder(model, Path("data/index/emb"))
        t = time.perf_counter()
        vectors = emb.embed_documents(texts)
        embed_s = time.perf_counter() - t
        name = f"bench_{corpus_digest(chunks, emb.name)}"
        store = QdrantStore(client, name, emb.dim)
        if client.collection_exists(store.collection):
            client.delete_collection(store.collection)
        store.ensure_collection()
        t = time.perf_counter()
        store.upsert(chunks, vectors)
        upsert_s = time.perf_counter() - t
        client.delete_collection(store.collection)
        t = time.perf_counter()
        for q in ("how do I add background tasks", "what is dependency injection", "cors"):
            emb.inner.embed_query(q)  # type: ignore[attr-defined]
        query_ms = (time.perf_counter() - t) / 3 * 1000
        cache_dir = Path("data/index/emb") / emb.name.replace(":", "_").replace("/", "_")
        out[model] = {
            "dim": emb.dim,
            "n_chunks": len(chunks),
            "cache_misses": emb.misses,  # type: ignore[attr-defined]
            "embed_s": embed_s,
            "upsert_s": upsert_s,
            "query_embed_ms_mean_of_3": query_ms,
            "vectors_bytes": int(vectors.nbytes),
            "cache_dir": cache_dir.as_posix(),
        }
        print(model, out[model], flush=True)
    run = make_run_dir(Path("experiments"), "m3_embed_bench")
    write_json(run / "bench.json", out)
    env = environment_info(command="scripts/m3_embed_bench.py", models=models)
    write_json(run / "env.json", env)
    print(f"-> {run}")


if __name__ == "__main__":
    main(sys.argv[1:])
