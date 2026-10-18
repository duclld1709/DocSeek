"""Sanity-check retrievers for the eval harness itself (PROJECT_SPEC §6 M1).

- ``RandomRetriever`` returns random readable chunks: metrics should be close to 0.
- ``OracleRetriever`` returns the golden relevant sections, best grade first: Recall@k
  should be exactly 1 whenever k >= |R|.

Both enforce ACL like real retrievers, so Leakage must be 0 for them too.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping

from docseek.core.acl import AccessPolicy
from docseek.core.models import Chunk, Query, ScoredHit


class RandomRetriever:
    """Uniformly random readable chunks; deterministic per ``(seed, qid)``."""

    def __init__(self, chunks: Iterable[Chunk], policy: AccessPolicy, *, seed: int = 13) -> None:
        self.name = f"random-{seed}"
        self._chunks = sorted(chunks, key=lambda c: c.chunk_id)
        self._policy = policy
        self._seed = seed

    def search(self, query: Query, k: int) -> list[ScoredHit]:
        visible = [
            c for c in self._chunks if self._policy.can_read(query.principal, c.metadata["acl"])
        ]
        digest = hashlib.sha256(f"{self._seed}:{query.qid}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        picked = rng.sample(visible, min(k, len(visible)))
        return [
            ScoredHit(
                chunk_id=c.chunk_id,
                section_id=c.section_id,
                score=1.0 / rank,
                rank=rank,
                retriever=self.name,
            )
            for rank, c in enumerate(picked, start=1)
        ]


class OracleRetriever:
    """Looks up the golden answer for ``query.qid``: grade-2 sections first, then grade 1."""

    name = "oracle"

    def __init__(
        self,
        relevant: Mapping[str, Mapping[str, int]],
        chunks: Iterable[Chunk],
        policy: AccessPolicy,
    ) -> None:
        self._relevant = relevant
        self._policy = policy
        self._first_chunk: dict[str, Chunk] = {}
        by_section: defaultdict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            by_section[c.section_id].append(c)
        for sid, cs in by_section.items():
            self._first_chunk[sid] = min(cs, key=lambda c: c.chunk_id)

    def search(self, query: Query, k: int) -> list[ScoredHit]:
        qrels = self._relevant.get(query.qid, {})
        ordered = sorted(qrels, key=lambda sid: (-qrels[sid], sid))
        hits: list[ScoredHit] = []
        for sid in ordered:
            chunk = self._first_chunk.get(sid)
            if chunk is None or not self._policy.can_read(query.principal, chunk.metadata["acl"]):
                continue
            rank = len(hits) + 1
            hits.append(
                ScoredHit(
                    chunk_id=chunk.chunk_id,
                    section_id=sid,
                    score=float(qrels[sid]),
                    rank=rank,
                    retriever=self.name,
                )
            )
            if len(hits) == k:
                break
        return hits
