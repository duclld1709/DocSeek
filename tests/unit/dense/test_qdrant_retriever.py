"""Dense retrieval on Qdrant ``:memory:`` with HashEmbedder: ACL, filters, idempotency."""

from pathlib import Path

import numpy as np
import pytest

from docseek.core.acl import AccessPolicy
from docseek.core.config import UnknownPrincipalError, load_principals
from docseek.core.models import MetadataFilter, Query
from docseek.dense.embedders import CachedEmbedder
from docseek.dense.qdrant_store import QdrantStore, connect, point_id
from docseek.dense.retriever import DenseRetriever, build_dense, index_chunks
from docseek.eval.golden import load_golden
from docseek.testing.acl_checks import probe_leaks
from docseek.testing.fakes import HashEmbedder
from docseek.testing.fixtures import MINI_GOLDEN, REPO_ROOT, mini_ingest

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))
CHUNKS = mini_ingest().chunks
CHUNK_ACL = {c.chunk_id: c.metadata["acl"] for c in CHUNKS}


@pytest.fixture(scope="module")
def retriever() -> DenseRetriever:
    return build_dense({"location": ":memory:"}, CHUNKS, POLICY, embedder=HashEmbedder(dim=128))


def test_no_leakage_for_any_principal(retriever: DenseRetriever) -> None:
    result = probe_leaks(retriever, CHUNK_ACL, POLICY)
    assert result.leaks == []
    assert result.restricted_seen["anonymous"] == 0
    assert result.restricted_seen["engineer"] > 0


def test_permission_queries_never_leak(retriever: DenseRetriever) -> None:
    perm = [q for q in load_golden(MINI_GOLDEN) if str(q.qtype) == "permission"]
    assert perm
    for q in perm:
        for principal in ("anonymous", "engineer", "security"):
            hits = retriever.search(Query(qid=q.qid, text=q.text, principal=principal), 10)
            assert all(POLICY.can_read(principal, CHUNK_ACL[h.chunk_id]) for h in hits)


def test_unknown_principal_fails_closed(retriever: DenseRetriever) -> None:
    with pytest.raises(UnknownPrincipalError):
        retriever.search(Query(qid="x", text="payment", principal="intruder"), 5)


def test_metadata_filter(retriever: DenseRetriever) -> None:
    q = Query(
        qid="x",
        text="Payment API timeouts",
        principal="security",
        filters=MetadataFilter(doc_type=["runbook"], status=["active"]),
    )
    hits = retriever.search(q, 10)
    by_id = {c.chunk_id: c for c in CHUNKS}
    assert hits
    for h in hits:
        meta = by_id[h.chunk_id].metadata
        assert meta["doc_type"] == "runbook" and meta["status"] == "active"


def test_upsert_is_idempotent() -> None:
    emb = HashEmbedder(dim=32)
    store = QdrantStore(connect(":memory:"), "c", emb.dim)
    assert index_chunks(CHUNKS, emb, store)[0] is True
    assert index_chunks(CHUNKS, emb, store)[0] is False  # complete: skipped
    store.upsert(CHUNKS[:5], emb.embed_documents([c.text for c in CHUNKS[:5]]))
    assert store.count() == len(CHUNKS)
    assert point_id("a") == point_id("a") != point_id("b")


def test_embedding_cache_round_trip(tmp_path: Path) -> None:
    emb = CachedEmbedder(HashEmbedder(dim=16), tmp_path)
    a = emb.embed_documents(["x", "y", "x"])
    assert emb.misses == 2 and emb.hits == 1
    again = CachedEmbedder(HashEmbedder(dim=16), tmp_path)
    b = again.embed_documents(["y", "x"])
    assert again.misses == 0 and np.allclose(b, a[[1, 0]])
    assert np.allclose(again.embed_query("x"), HashEmbedder(dim=16).embed_query("x"))
