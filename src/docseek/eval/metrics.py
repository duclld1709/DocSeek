"""Retrieval metrics, implemented from their definitions (PROJECT_SPEC §5.5).

All ranking metrics take ``ranked``: section ids, best first, already de-duplicated (use
:func:`hits_to_sections` for chunk-level hits), and ``qrels``: ``section_id -> grade`` with
grade 2 = answers directly, 1 = related/partial. Sections absent from ``qrels`` are grade 0.
A section counts as relevant when its grade is >= 1.

Conventions (cross-checked against ``ranx`` in tests):

- nDCG uses exponential gain ``2^rel - 1`` and ``log2(i + 1)`` discount; that is ranx's
  ``ndcg_burges``, not its ``ndcg`` (linear gain).
- AP@k divides by ``min(|R|, k)`` as in the spec. ranx divides by ``|R|``; the two agree
  whenever ``|R| <= k``.
- Queries with no relevant section (unanswerable ones) are undefined for these metrics;
  callers must exclude them from averages instead of scoring them 0.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from docseek.core.models import ScoredHit

Qrels = Mapping[str, int]


def hits_to_sections(hits: Iterable[ScoredHit]) -> list[str]:
    """Section ids in rank order, keeping each section's best (first) rank only.

    Hits are sorted by ``rank`` first so callers need not pre-sort.
    """
    seen: set[str] = set()
    out: list[str] = []
    for hit in sorted(hits, key=lambda h: h.rank):
        if hit.section_id not in seen:
            seen.add(hit.section_id)
            out.append(hit.section_id)
    return out


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")


def _relevant(qrels: Qrels) -> set[str]:
    return {sid for sid, grade in qrels.items() if grade > 0}


def recall_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    """|R ∩ top_k| / |R|."""
    _check_k(k)
    rel = _relevant(qrels)
    if not rel:
        raise ValueError("recall is undefined for a query without relevant sections")
    return len(rel.intersection(ranked[:k])) / len(rel)


def precision_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    """|R ∩ top_k| / k (missing positions count as non-relevant)."""
    _check_k(k)
    return len(_relevant(qrels).intersection(ranked[:k])) / k


def mrr_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    """1 / rank of the first relevant section within the top k, else 0."""
    _check_k(k)
    rel = _relevant(qrels)
    for i, sid in enumerate(ranked[:k], start=1):
        if sid in rel:
            return 1.0 / i
    return 0.0


def hit_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    """1 if any relevant section is in the top k, else 0."""
    _check_k(k)
    return 1.0 if _relevant(qrels).intersection(ranked[:k]) else 0.0


def average_precision_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    """Mean of precision@i over relevant positions i <= k, divided by min(|R|, k)."""
    _check_k(k)
    rel = _relevant(qrels)
    if not rel:
        raise ValueError("AP is undefined for a query without relevant sections")
    found = 0
    total = 0.0
    for i, sid in enumerate(ranked[:k], start=1):
        if sid in rel:
            found += 1
            total += found / i
    return total / min(len(rel), k)


def dcg_at_k(grades: Sequence[int], k: int) -> float:
    """Σ (2^rel_i - 1) / log2(i + 1) over the first k grades."""
    _check_k(k)
    return sum((2.0**g - 1.0) / math.log2(i + 1) for i, g in enumerate(grades[:k], start=1))


def ndcg_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    """DCG@k / IDCG@k with exponential gains."""
    _check_k(k)
    ideal = dcg_at_k(sorted((g for g in qrels.values() if g > 0), reverse=True), k)
    if ideal == 0:
        raise ValueError("nDCG is undefined for a query without relevant sections")
    return dcg_at_k([qrels.get(sid, 0) for sid in ranked], k) / ideal


def leakage_at_k(
    ranked: Sequence[str],
    section_acl: Mapping[str, Iterable[str]],
    allowed_groups: Iterable[str],
    k: int,
) -> int:
    """Number of top-k sections the principal may not read. Must always be 0.

    A section with no known ACL counts as a leak (fail closed).
    """
    _check_k(k)
    allowed = set(allowed_groups)
    return sum(1 for sid in ranked[:k] if not allowed.intersection(section_acl.get(sid, ())))


def stale_hit_at_k(
    ranked: Sequence[str],
    stale: Iterable[str],
    fresh: Iterable[str],
    k: int,
) -> float:
    """1 if a stale section (deprecated or superseded) ranks above every fresh one in top k.

    ``fresh`` are the sections of the newest document(s) that answer the query. If no fresh
    section is in the top k, any stale section in the top k counts as a stale hit.
    """
    _check_k(k)
    stale_set, fresh_set = set(stale), set(fresh)
    for sid in ranked[:k]:
        if sid in fresh_set:
            return 0.0
        if sid in stale_set:
            return 1.0
    return 0.0


def abstention_accuracy(abstained: Sequence[bool]) -> float:
    """Share of unanswerable queries the system correctly declined to answer."""
    if not abstained:
        raise ValueError("no unanswerable queries")
    return sum(abstained) / len(abstained)


RANKING_METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": mrr_at_k,
    "map": average_precision_at_k,
    "ndcg": ndcg_at_k,
    "hit": hit_at_k,
}
