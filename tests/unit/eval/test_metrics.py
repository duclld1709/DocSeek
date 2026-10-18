"""Metrics: hand-computed examples, edge cases and property-based invariants."""

import math
import random
from collections.abc import Callable, Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from docseek.core.models import ScoredHit
from docseek.eval.metrics import (
    Qrels,
    abstention_accuracy,
    average_precision_at_k,
    dcg_at_k,
    hit_at_k,
    hits_to_sections,
    leakage_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    stale_hit_at_k,
)

Metric = Callable[[Sequence[str], Qrels, int], float]

# Worked example used throughout: R = {a: 2, b: 1, c: 1}, ranking a x b y c z.
RANKED = ["a", "x", "b", "y", "c", "z"]
QRELS = {"a": 2, "b": 1, "c": 1}


def test_recall_by_hand() -> None:
    assert recall_at_k(RANKED, QRELS, 1) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, QRELS, 3) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, QRELS, 10) == 1.0


def test_precision_by_hand() -> None:
    assert precision_at_k(RANKED, QRELS, 2) == 0.5
    assert precision_at_k(RANKED, QRELS, 5) == pytest.approx(3 / 5)
    assert precision_at_k(RANKED, QRELS, 10) == pytest.approx(3 / 10)  # short lists padded


def test_mrr_and_hit_by_hand() -> None:
    assert mrr_at_k(["x", "y", "b"], QRELS, 3) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "b"], QRELS, 2) == 0.0
    assert hit_at_k(["x", "y", "b"], QRELS, 3) == 1.0
    assert hit_at_k(["x", "y", "b"], QRELS, 2) == 0.0


def test_average_precision_by_hand() -> None:
    # relevant at 1, 3, 5: (1/1 + 2/3 + 3/5) / min(3, k)
    assert average_precision_at_k(RANKED, QRELS, 10) == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)
    # k=2: only position 1 counts, divided by min(3, 2) = 2
    assert average_precision_at_k(RANKED, QRELS, 2) == pytest.approx(1 / 2)


def test_ndcg_by_hand() -> None:
    dcg = 3 / math.log2(2) + 1 / math.log2(4) + 1 / math.log2(6)
    idcg = 3 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, QRELS, 6) == pytest.approx(dcg / idcg)
    assert dcg_at_k([2, 0, 1], 3) == pytest.approx(3 + 0.5)
    assert ndcg_at_k(["a", "b", "c"], QRELS, 3) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", [recall_at_k, average_precision_at_k, ndcg_at_k])
def test_undefined_metrics_raise_without_relevant_sections(metric: Metric) -> None:
    with pytest.raises(ValueError, match="undefined"):
        metric(RANKED, {"a": 0}, 5)


@pytest.mark.parametrize("metric", [recall_at_k, precision_at_k, mrr_at_k, hit_at_k, ndcg_at_k])
def test_k_must_be_positive(metric: Metric) -> None:
    with pytest.raises(ValueError, match="k must be"):
        metric(RANKED, QRELS, 0)


def _hit(chunk: str, section: str, rank: int) -> ScoredHit:
    return ScoredHit(chunk_id=chunk, section_id=section, score=1 / rank, rank=rank, retriever="t")


def test_hits_to_sections_keeps_best_rank_per_section() -> None:
    hits = [_hit("b0", "b", 3), _hit("a1", "a", 2), _hit("a0", "a", 1), _hit("c0", "c", 4)]
    assert hits_to_sections(hits) == ["a", "b", "c"]


def test_leakage_counts_forbidden_and_unknown_sections() -> None:
    acl = {"pub": ["public"], "eng": ["eng-internal"], "sec": ["security-team"]}
    ranked = ["pub", "eng", "sec", "ghost"]
    assert leakage_at_k(ranked, acl, ["public"], 4) == 3  # eng, sec and the unknown ghost
    assert leakage_at_k(ranked, acl, ["public", "eng-internal"], 2) == 0
    assert leakage_at_k(ranked, acl, ["public", "eng-internal", "security-team"], 3) == 0


@pytest.mark.parametrize(
    ("ranked", "expected"),
    [
        (["new", "old"], 0.0),
        (["old", "new"], 1.0),
        (["x", "old"], 1.0),  # stale present, fresh missing
        (["x", "y"], 0.0),
        (["x", "new", "old"], 0.0),
    ],
)
def test_stale_hit(ranked: list[str], expected: float) -> None:
    assert stale_hit_at_k(ranked, stale=["old"], fresh=["new"], k=3) == expected


def test_abstention_accuracy() -> None:
    assert abstention_accuracy([True, False, True, True]) == 0.75
    with pytest.raises(ValueError):
        abstention_accuracy([])


# --- properties ---------------------------------------------------------------------------

ids = st.sampled_from([f"s{i}" for i in range(12)])
rankings = st.lists(ids, unique=True, max_size=12)
qrels_st = st.dictionaries(ids, st.integers(1, 2), min_size=1, max_size=6)
ks = st.integers(1, 15)
ALL: list[Metric] = [
    recall_at_k,
    precision_at_k,
    mrr_at_k,
    hit_at_k,
    average_precision_at_k,
    ndcg_at_k,
]


@given(rankings, qrels_st, ks)
def test_metrics_are_bounded(ranked: list[str], qrels: dict[str, int], k: int) -> None:
    for metric in ALL:
        assert -1e-12 <= metric(ranked, qrels, k) <= 1.0 + 1e-12


@given(rankings, qrels_st, ks, st.randoms(use_true_random=False))
def test_permuting_non_relevant_results_changes_nothing(
    ranked: list[str], qrels: dict[str, int], k: int, rnd: random.Random
) -> None:
    slots = [i for i, sid in enumerate(ranked) if sid not in qrels]
    values = [ranked[i] for i in slots]
    rnd.shuffle(values)
    permuted = list(ranked)
    for i, sid in zip(slots, values, strict=True):
        permuted[i] = sid
    for metric in ALL:
        assert metric(permuted, qrels, k) == pytest.approx(metric(ranked, qrels, k))


@given(qrels_st, ks)
def test_ideal_ranking_scores_one(qrels: dict[str, int], k: int) -> None:
    ideal = sorted(qrels, key=lambda s: -qrels[s])
    assert ndcg_at_k(ideal, qrels, k) == pytest.approx(1.0)
    assert average_precision_at_k(ideal, qrels, k) == pytest.approx(1.0)
    assert mrr_at_k(ideal, qrels, k) == 1.0


@given(rankings, qrels_st, st.integers(1, 14))
def test_recall_is_monotone_in_k(ranked: list[str], qrels: dict[str, int], k: int) -> None:
    assert recall_at_k(ranked, qrels, k) <= recall_at_k(ranked, qrels, k + 1)


@given(rankings, qrels_st, ks)
def test_appending_irrelevant_results_changes_nothing(
    ranked: list[str], qrels: dict[str, int], k: int
) -> None:
    longer = [*ranked, "zz-irrelevant-1", "zz-irrelevant-2"]
    for metric in ALL:
        assert metric(longer, qrels, k) == pytest.approx(metric(ranked, qrels, k))
