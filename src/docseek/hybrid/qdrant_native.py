"""Hybrid search inside Qdrant: a BM25 sparse vector next to the dense one, fused with the
Query API (``prefetch`` both, then RRF or DBSF) — the comparison point for our own fusion.

The sparse vector holds BM25's document side, ``tf * (k1 + 1) / (tf + k1 * B(d))``; Qdrant's
``idf`` modifier supplies ``ln(1 + (N - df + 0.5) / (df + 0.5))`` at query time, which is the
same Lucene idf our scorer uses. Query vectors are the unique query terms with weight 1.
Term ids are a stable 31-bit hash of the term.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from docseek.core.acl import AccessPolicy
from docseek.core.interfaces import Embedder
from docseek.core.models import Chunk, Query, ScoredHit
from docseek.dense.qdrant_store import DENSE, KEYWORD_FIELDS, point_id, to_qdrant_filter
from docseek.lexical.inverted_index import ALL_FIELD, chunk_fields
from docseek.lexical.tokenizer import Tokenizer, TokenizerConfig

SPARSE = "bm25"


def term_id(term: str) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode(), digest_size=4).digest(), "big") >> 1


def bm25_doc_vectors(
    term_lists: Sequence[list[str]], *, k1: float, b: float
) -> list[models.SparseVector]:
    lengths = [len(t) for t in term_lists]
    avg = sum(lengths) / len(lengths) if lengths else 0.0
    out: list[models.SparseVector] = []
    for terms, dl in zip(term_lists, lengths, strict=True):
        norm = (1 - b) + b * (dl / avg if avg else 1.0)
        weights: dict[int, float] = {}
        for term, tf in Counter(terms).items():
            weights[term_id(term)] = tf * (k1 + 1) / (tf + k1 * norm)
        out.append(models.SparseVector(indices=list(weights), values=list(weights.values())))
    return out


class QdrantHybridRetriever:
    def __init__(
        self,
        name: str,
        client: QdrantClient,
        collection: str,
        embedder: Embedder,
        policy: AccessPolicy,
        *,
        tokenizer: TokenizerConfig,
        fusion: str = "rrf",
        depth: int = 50,
        rrf_k: int | None = None,
    ) -> None:
        self.name = name
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._policy = policy
        self._tokenizer = Tokenizer(tokenizer)
        self._fusion = fusion
        self._depth = depth
        self._rrf_k = rrf_k

    def build(self, chunks: Sequence[Chunk], *, k1: float, b: float) -> None:
        """Create and fill the collection unless it already holds every chunk."""
        c = self._client
        if c.collection_exists(self._collection):
            if c.count(self._collection, exact=True).count == len(chunks):
                return
            c.delete_collection(self._collection)
        c.create_collection(
            self._collection,
            vectors_config={
                DENSE: models.VectorParams(size=self._embedder.dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for name in KEYWORD_FIELDS:
            c.create_payload_index(self._collection, name, models.PayloadSchemaType.KEYWORD)
        dense = self._embedder.embed_documents([ch.text_for_embedding for ch in chunks])
        terms = [self._tokenizer.tokenize(chunk_fields(ch)[ALL_FIELD]) for ch in chunks]
        sparse = bm25_doc_vectors(terms, k1=k1, b=b)
        for start in range(0, len(chunks), 256):
            c.upsert(
                self._collection,
                points=[
                    models.PointStruct(
                        id=point_id(ch.chunk_id),
                        vector={DENSE: dense[i].tolist(), SPARSE: sparse[i]},
                        payload={
                            **ch.metadata,
                            "chunk_id": ch.chunk_id,
                            "section_id": ch.section_id,
                        },
                    )
                    for i, ch in enumerate(chunks[start : start + 256], start=start)
                ],
                wait=True,
            )

    def search(self, query: Query, k: int) -> list[ScoredHit]:
        groups = self._policy.groups(query.principal)  # unknown principal raises
        flt = to_qdrant_filter(groups, query.filters)
        terms = list(dict.fromkeys(self._tokenizer.tokenize(query.text)))
        sparse_q = models.SparseVector(
            indices=[term_id(t) for t in terms], values=[1.0] * len(terms)
        )
        dense_q = self._embedder.embed_query(query.text).tolist()
        fusion: models.FusionQuery | models.RrfQuery
        if self._fusion == "rrf" and self._rrf_k is not None:
            fusion = models.RrfQuery(rrf=models.Rrf(k=self._rrf_k))
        else:
            fusion = models.FusionQuery(
                fusion=models.Fusion.RRF if self._fusion == "rrf" else models.Fusion.DBSF
            )
        res = self._client.query_points(
            self._collection,
            prefetch=[
                models.Prefetch(query=sparse_q, using=SPARSE, filter=flt, limit=self._depth),
                models.Prefetch(query=dense_q, using=DENSE, filter=flt, limit=self._depth),
            ],
            query=fusion,
            query_filter=flt,
            limit=k,
            with_payload=["chunk_id", "section_id"],
        )
        return [
            ScoredHit(
                chunk_id=str((p.payload or {})["chunk_id"]),
                section_id=str((p.payload or {})["section_id"]),
                score=float(p.score),
                rank=i,
                retriever=self.name,
            )
            for i, p in enumerate(res.points, start=1)
        ]


def collection_name(prefix: str, *parts: str) -> str:
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, '|'.join(parts)).hex[:12]}"
