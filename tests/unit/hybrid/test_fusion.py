"""RRF, weighted fusion and normalisation against hand-computed values and edge cases."""

import pytest

from docseek.core.models import ScoredHit
from docseek.hybrid.fusion import rrf, weighted
from docseek.hybrid.normalization import min_max, z_score


def _hits(*ids: str, scores: list[float] | None = None) -> list[ScoredHit]:
    scores = scores or [float(len(ids) - i) for i in range(len(ids))]
    return [
        ScoredHit(chunk_id=c, section_id=f"s-{c}", score=s, rank=i, retriever="x")
        for i, (c, s) in enumerate(zip(ids, scores, strict=True), start=1)
    ]


def test_rrf_hand_computed() -> None:
    # k=60. a: 1/61 + 1/63; b: 1/62 + 1/61; c: 1/63; d: 1/62.
    out = rrf({"bm25": _hits("a", "b", "c"), "dense": _hits("b", "d", "a")}, k=10, rrf_k=60)
    scores = {h.chunk_id: h.score for h in out}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)
    assert scores["d"] == pytest.approx(1 / 62)
    assert [h.chunk_id for h in out] == ["b", "a", "d", "c"]
    assert [h.rank for h in out] == [1, 2, 3, 4]
    assert out[0].debug == {"rank_bm25": 2, "rank_dense": 1}


def test_rrf_edge_cases() -> None:
    assert rrf({}, k=5) == []
    assert rrf({"a": [], "b": []}, k=5) == []
    # Duplicate id in one list: only its best rank counts.
    dup = [
        *_hits("a", "b"),
        ScoredHit(chunk_id="a", section_id="s-a", score=0, rank=3, retriever="x"),
    ]
    out = rrf({"x": dup}, k=5, rrf_k=60)
    assert [h.chunk_id for h in out] == ["a", "b"]
    assert out[0].score == pytest.approx(1 / 61)
    # Tie (same fused score): best single rank wins, then chunk id.
    tie = rrf({"x": _hits("p", "q"), "y": _hits("q", "p")}, k=5)
    assert tie[0].score == pytest.approx(tie[1].score)
    assert [h.chunk_id for h in tie] == ["p", "q"]
    # k truncates.
    assert len(rrf({"x": _hits("a", "b", "c")}, k=2)) == 2


def test_normalisation_hand_computed() -> None:
    assert min_max([3.0, 1.0, 2.0]) == [1.0, 0.0, 0.5]
    assert min_max([2.0, 2.0]) == [1.0, 1.0]
    assert min_max([]) == []
    z = z_score([1.0, 2.0, 3.0])  # mean 2, population std sqrt(2/3)
    assert z == pytest.approx([-1.224745, 0.0, 1.224745], abs=1e-6)
    assert z_score([5.0, 5.0]) == [0.0, 0.0]


def test_weighted_minmax_hand_computed() -> None:
    # bm25 scores 10,6,2 -> 1, .5, 0 ; dense .9,.8 -> 1, 0 ; alpha=.3 on bm25.
    lists = {
        "bm25": _hits("a", "b", "c", scores=[10, 6, 2]),
        "dense": _hits("b", "d", scores=[0.9, 0.8]),
    }
    out = weighted(lists, {"bm25": 0.3, "dense": 0.7}, k=10)
    s = {h.chunk_id: h.score for h in out}
    assert s["a"] == pytest.approx(0.3 * 1 + 0.7 * 0)  # missing in dense -> floor 0
    assert s["b"] == pytest.approx(0.3 * 0.5 + 0.7 * 1)
    assert s["c"] == pytest.approx(0.0)
    assert s["d"] == pytest.approx(0.7 * 0)
    assert out[0].chunk_id == "b"
    with pytest.raises(ValueError, match="do not match"):
        weighted(lists, {"bm25": 1.0}, k=5)


def test_weighted_zscore_missing_gets_list_minimum() -> None:
    lists = {"x": _hits("a", "b", "c", scores=[3, 2, 1]), "y": _hits("c", scores=[1.0])}
    out = weighted(lists, {"x": 1.0, "y": 1.0}, k=5, norm="zscore")
    s = {h.chunk_id: h.score for h in out}
    # y has one item -> z = 0 for c and floor 0 for a, b.
    assert s["a"] == pytest.approx(1.224745, abs=1e-6)
    assert s["c"] == pytest.approx(-1.224745, abs=1e-6)
