"""Rank fusion of several retrievers' hit lists (keyed by ``chunk_id``).

- **RRF**: ``score(d) = sum_i 1 / (k + rank_i(d))`` over the lists containing ``d``.
- **Weighted sum**: normalise each list's scores, then ``sum_i w_i * norm_i(d)``. A chunk
  missing from list ``i`` gets that list's minimum normalised score (0 for min-max), i.e. it
  is treated as no better than the worst hit retrieved.

Ties are broken by the best rank the chunk reached in any list, then by ``chunk_id``, so the
output is deterministic. Duplicate chunk ids within one list keep their first (best) entry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from docseek.core.models import Query, ScoredHit
from docseek.hybrid.normalization import NORMALIZERS


def _dedupe(hits: Sequence[ScoredHit]) -> list[ScoredHit]:
    seen: set[str] = set()
    out: list[ScoredHit] = []
    for h in sorted(hits, key=lambda h: h.rank):
        if h.chunk_id not in seen:
            seen.add(h.chunk_id)
            out.append(h)
    return out


def _finish(
    scores: dict[str, float],
    best_rank: dict[str, int],
    section: dict[str, str],
    debug: dict[str, dict[str, Any]],
    k: int,
    name: str,
) -> list[ScoredHit]:
    order = sorted(scores, key=lambda c: (-scores[c], best_rank[c], c))[:k]
    return [
        ScoredHit(
            chunk_id=c,
            section_id=section[c],
            score=scores[c],
            rank=i,
            retriever=name,
            debug=debug[c],
        )
        for i, c in enumerate(order, start=1)
    ]


def rrf(
    lists: Mapping[str, Sequence[ScoredHit]], *, k: int, rrf_k: float = 60.0, name: str = "rrf"
) -> list[ScoredHit]:
    """Reciprocal rank fusion; ``lists`` maps a retriever label to its hits."""
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    section: dict[str, str] = {}
    debug: dict[str, dict[str, Any]] = {}
    for label, hits in lists.items():
        for rank, h in enumerate(_dedupe(hits), start=1):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            best[h.chunk_id] = min(best.get(h.chunk_id, rank), rank)
            section[h.chunk_id] = h.section_id
            debug.setdefault(h.chunk_id, {})[f"rank_{label}"] = rank
    return _finish(scores, best, section, debug, k, name)


def weighted(
    lists: Mapping[str, Sequence[ScoredHit]],
    weights: Mapping[str, float],
    *,
    k: int,
    norm: str = "minmax",
    name: str = "weighted",
) -> list[ScoredHit]:
    """Weighted sum of normalised scores; ``weights`` keys must match ``lists`` keys."""
    if set(weights) != set(lists):
        raise ValueError(f"weights {sorted(weights)} do not match lists {sorted(lists)}")
    normalise = NORMALIZERS[norm]
    per_list: dict[str, dict[str, float]] = {}
    floor: dict[str, float] = {}
    best: dict[str, int] = {}
    section: dict[str, str] = {}
    debug: dict[str, dict[str, Any]] = {}
    for label, hits in lists.items():
        deduped = _dedupe(hits)
        normed = normalise([h.score for h in deduped])
        per_list[label] = {h.chunk_id: s for h, s in zip(deduped, normed, strict=True)}
        floor[label] = min(normed) if normed else 0.0
        for rank, h in enumerate(deduped, start=1):
            best[h.chunk_id] = min(best.get(h.chunk_id, rank), rank)
            section[h.chunk_id] = h.section_id
            debug.setdefault(h.chunk_id, {})[f"rank_{label}"] = rank
    scores = {
        c: sum(w * per_list[label].get(c, floor[label]) for label, w in weights.items())
        for c in section
    }
    return _finish(scores, best, section, debug, k, name)


class HybridRetriever:
    """Runs each child retriever to ``depth`` and fuses their lists.

    ACL is enforced by every child; fusion only reorders hits they already allowed.
    """

    def __init__(
        self,
        name: str,
        retrievers: Mapping[str, Any],
        *,
        method: str = "rrf",
        depth: int = 50,
        rrf_k: float = 60.0,
        weights: Mapping[str, float] | None = None,
        norm: str = "minmax",
    ) -> None:
        if method not in ("rrf", "weighted"):
            raise ValueError(f"unknown fusion method {method!r}")
        if method == "weighted" and not weights:
            raise ValueError("weighted fusion needs weights")
        self.name = name
        self._retrievers = dict(retrievers)
        self._method = method
        self._depth = depth
        self._rrf_k = rrf_k
        self._weights = dict(weights or {})
        self._norm = norm

    def child_hits(self, query: Query) -> dict[str, list[ScoredHit]]:
        return {label: r.search(query, self._depth) for label, r in self._retrievers.items()}

    def fuse(self, lists: Mapping[str, Sequence[ScoredHit]], k: int) -> list[ScoredHit]:
        if self._method == "rrf":
            return rrf(lists, k=k, rrf_k=self._rrf_k, name=self.name)
        return weighted(lists, self._weights, k=k, norm=self._norm, name=self.name)

    def search(self, query: Query, k: int) -> list[ScoredHit]:
        return self.fuse(self.child_hits(query), k)
