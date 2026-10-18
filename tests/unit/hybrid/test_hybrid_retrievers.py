"""Hybrid retrievers on the fixture corpora: Leakage = 0, Qdrant-native BM25 matches ours."""

import math

import pytest

from docseek.core.acl import AccessPolicy
from docseek.core.config import load_principals
from docseek.core.models import Query
from docseek.dense.qdrant_store import connect
from docseek.dense.retriever import build_dense
from docseek.hybrid.fusion import HybridRetriever
from docseek.hybrid.qdrant_native import QdrantHybridRetriever, bm25_doc_vectors
from docseek.lexical import factory
from docseek.lexical.tokenizer import TokenizerConfig
from docseek.rerank.cross_encoder import RerankedRetriever, rerank_texts
from docseek.testing.acl_checks import probe_leaks
from docseek.testing.fakes import HashEmbedder, IdentityReranker
from docseek.testing.fixtures import REPO_ROOT, mini_ingest

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))
CHUNKS = mini_ingest().chunks
CHUNK_ACL = {c.chunk_id: c.metadata["acl"] for c in CHUNKS}


@pytest.fixture(scope="module")
def children() -> dict:
    return {
        "bm25": factory.build_bm25({"index_dir": None}, CHUNKS, POLICY),
        "dense": build_dense(
            {"location": ":memory:"}, CHUNKS, POLICY, embedder=HashEmbedder(dim=64)
        ),
    }


@pytest.fixture(scope="module")
def native() -> QdrantHybridRetriever:
    r = QdrantHybridRetriever(
        "qdrant-rrf",
        connect(":memory:"),
        "hyb",
        HashEmbedder(dim=64),
        POLICY,
        tokenizer=TokenizerConfig(),
    )
    r.build(CHUNKS, k1=1.2, b=0.75)
    return r


@pytest.mark.parametrize("method", ["rrf", "weighted"])
def test_hybrid_never_leaks(children: dict, method: str) -> None:
    r = HybridRetriever("h", children, method=method, weights={"bm25": 0.5, "dense": 0.5}, depth=20)
    result = probe_leaks(r, CHUNK_ACL, POLICY)
    assert result.leaks == [] and result.restricted_seen["engineer"] > 0


def test_reranked_never_leaks(children: dict) -> None:
    r = RerankedRetriever("rr", children["bm25"], IdentityReranker(), rerank_texts(CHUNKS), top_n=5)
    result = probe_leaks(r, CHUNK_ACL, POLICY)
    assert result.leaks == [] and result.restricted_seen["engineer"] > 0
    assert len(r.search(Query(qid="x", text="dependencies"), 3)) == 3


def test_qdrant_native_never_leaks(native: QdrantHybridRetriever) -> None:
    result = probe_leaks(native, CHUNK_ACL, POLICY)
    assert result.leaks == [] and result.restricted_seen["engineer"] > 0


def test_sparse_bm25_doc_weights() -> None:
    # dl = avgdl -> B = 1 -> weight = tf*(k1+1)/(tf+k1); tf=2, k1=1.2 -> 4.4/3.2
    (v, _) = bm25_doc_vectors([["a", "a", "b"], ["c", "c", "c"]], k1=1.2, b=0.75)
    weights = sorted(v.values)
    assert weights[1] == pytest.approx(2 * 2.2 / (2 + 1.2))
    assert weights[0] == pytest.approx(2.2 / (1 + 1.2))
    assert all(math.isfinite(w) for w in weights)
