"""Metadata as a third signal: freshness boost and penalties for stale documents.

For a hit with score ``s`` from a doc updated ``Δ`` days before ``reference``::

    s' = s * (1 + beta * exp(-Δ / tau)) * (1 - deprecated_penalty)?  * (1 - superseded_penalty)?

The penalties apply when the chunk's ``status`` is ``deprecated`` and when another document
``supersedes`` its document. Hits are re-sorted by ``s'``. Scores must be non-negative
(BM25, cosine of normalised vectors on this corpus, RRF), otherwise boosting would invert.
``reference`` is fixed (newest ``updated_at`` in the corpus) so runs are reproducible.
Cross-encoder logits can be negative, so ``score_transform="sigmoid"`` maps them to (0, 1)
before the multiplier is applied.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from docseek.core.filtering import as_datetime
from docseek.core.interfaces import Retriever
from docseek.core.models import Chunk, Query, ScoredHit


@dataclass(frozen=True)
class FreshnessConfig:
    beta: float = 0.3
    tau_days: float = 180.0
    deprecated_penalty: float = 0.5
    superseded_penalty: float = 0.5


def superseded_docs(chunks: Sequence[Chunk]) -> frozenset[str]:
    """Doc ids that some other document declares it supersedes."""
    return frozenset(str(c.metadata["supersedes"]) for c in chunks if c.metadata.get("supersedes"))


def multiplier(
    updated_at: datetime | None,
    status: str,
    superseded: bool,
    reference: datetime,
    cfg: FreshnessConfig,
) -> float:
    m = 1.0
    if updated_at is not None:
        age = max((reference - updated_at).total_seconds() / 86400.0, 0.0)
        m *= 1.0 + cfg.beta * math.exp(-age / cfg.tau_days)
    if status == "deprecated":
        m *= 1.0 - cfg.deprecated_penalty
    if superseded:
        m *= 1.0 - cfg.superseded_penalty
    return m


class FreshnessReranker:
    """Wraps a retriever; over-fetches ``depth`` hits, rescales and re-sorts them."""

    def __init__(
        self,
        name: str,
        inner: Retriever,
        chunks: Sequence[Chunk],
        cfg: FreshnessConfig,
        *,
        depth: int = 50,
        score_transform: str = "identity",
    ) -> None:
        if score_transform not in ("identity", "sigmoid"):
            raise ValueError(f"unknown score_transform {score_transform!r}")
        self._sigmoid = score_transform == "sigmoid"
        self.name = name
        self._inner = inner
        self._cfg = cfg
        self._depth = depth
        self._meta = {c.chunk_id: c.metadata for c in chunks}
        self._superseded = superseded_docs(chunks)
        dates = [d for c in chunks if (d := as_datetime(c.metadata.get("updated_at")))]
        self.reference = max(dates) if dates else datetime.now().astimezone()

    def rescore(self, hits: Sequence[ScoredHit], k: int) -> list[ScoredHit]:
        scored: list[tuple[float, int, ScoredHit]] = []
        for h in hits:
            meta = self._meta.get(h.chunk_id, {})
            m = multiplier(
                as_datetime(meta.get("updated_at")),
                str(meta.get("status", "active")),
                meta.get("doc_id") in self._superseded,
                self.reference,
                self._cfg,
            )
            base = 1.0 / (1.0 + math.exp(-h.score)) if self._sigmoid else h.score
            scored.append((base * m, h.rank, h))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [
            h.model_copy(
                update={
                    "score": s,
                    "rank": i,
                    "retriever": self.name,
                    "debug": {**h.debug, "pre_freshness_rank": h.rank},
                }
            )
            for i, (s, _, h) in enumerate(scored[:k], start=1)
        ]

    def search(self, query: Query, k: int) -> list[ScoredHit]:
        return self.rescore(self._inner.search(query, max(k, self._depth)), k)
