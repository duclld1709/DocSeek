"""Dense retriever over a Qdrant collection, plus the builder used by the eval runner.

Options (experiment YAML ``retriever:`` block)::

    type: dense
    model: BAAI/bge-small-en-v1.5      # from configs/models.yaml candidates
    location: http://127.0.0.1:6333    # or ":memory:" / a directory
    cache_dir: data/index/emb          # embedding cache; null disables
    hnsw_ef: null                      # search-time ef; null = Qdrant default
    embedder: hash                     # optional: deterministic HashEmbedder (CI gate, demo)
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from docseek.core.acl import AccessPolicy
from docseek.core.interfaces import Embedder
from docseek.core.models import Chunk, Query, ScoredHit
from docseek.dense.qdrant_store import QdrantStore, connect


class DenseRetriever:
    def __init__(
        self,
        name: str,
        embedder: Embedder,
        store: QdrantStore,
        policy: AccessPolicy,
        *,
        hnsw_ef: int | None = None,
        exact: bool = False,
    ) -> None:
        self.name = name
        self._embedder = embedder
        self._store = store
        self._policy = policy
        self._hnsw_ef = hnsw_ef
        self._exact = exact

    def search(self, query: Query, k: int) -> list[ScoredHit]:
        groups = self._policy.groups(query.principal)  # unknown principal raises: fail closed
        vector = self._embedder.embed_query(query.text)
        rows = self._store.search(
            vector, k, groups=groups, flt=query.filters, hnsw_ef=self._hnsw_ef, exact=self._exact
        )
        return [
            ScoredHit(chunk_id=cid, section_id=sid, score=score, rank=i, retriever=self.name)
            for i, (cid, sid, score) in enumerate(rows, start=1)
        ]


def corpus_digest(chunks: Sequence[Chunk], embedder_name: str) -> str:
    h = hashlib.sha256(embedder_name.encode())
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(hashlib.sha256(c.text_for_embedding.encode()).digest())
        h.update(repr(sorted(c.metadata.items())).encode())
    return h.hexdigest()[:10]


def index_chunks(
    chunks: Sequence[Chunk], embedder: Embedder, store: QdrantStore
) -> tuple[bool, float]:
    """Embed and upsert unless the collection is already complete; returns (built, seconds)."""
    import time

    store.ensure_collection()
    if store.count() == len(chunks):
        return False, 0.0
    start = time.perf_counter()
    vectors = embedder.embed_documents([c.text_for_embedding for c in chunks])
    store.upsert(chunks, vectors)
    return True, time.perf_counter() - start


def make_embedder(model: str, cache_dir: Path | None) -> Embedder:
    from docseek.core.config import load_models_config
    from docseek.dense.embedders import CachedEmbedder, FastEmbedEmbedder

    prefixes = load_models_config().embedding.prefixes_for(model)
    inner = FastEmbedEmbedder(model, query_prefix=prefixes.query, passage_prefix=prefixes.passage)
    return CachedEmbedder(inner, cache_dir) if cache_dir else inner


def build_dense(
    opts: Mapping[str, Any],
    chunks: Sequence[Chunk],
    policy: AccessPolicy,
    *,
    embedder: Embedder | None = None,
) -> DenseRetriever:
    from docseek.core.config import Settings, load_models_config

    model = str(opts.get("model") or load_models_config().embedding.default)
    cache = opts.get("cache_dir", "data/index/emb")
    if embedder is None and opts.get("embedder") == "hash":
        from docseek.testing.fakes import HashEmbedder

        embedder = HashEmbedder(dim=int(opts.get("dim", 128)))
        model = embedder.name
    emb = embedder or make_embedder(model, Path(cache) if cache else None)
    location = str(opts.get("location") or Settings().qdrant_url)
    slug = re.sub(r"[^a-z0-9]+", "_", emb.name.lower()).strip("_")
    store = QdrantStore(
        connect(location), f"chunks_{slug}_{corpus_digest(chunks, emb.name)}", emb.dim
    )
    index_chunks(chunks, emb, store)
    ef = opts.get("hnsw_ef")
    return DenseRetriever(
        f"dense-{model.split('/')[-1]}",
        emb,
        store,
        policy,
        hnsw_ef=int(ef) if ef else None,
        exact=bool(opts.get("exact", False)),
    )
