"""Cross-check of every ranking metric against ranx on randomised runs (|Δ| < 1e-6)."""

import random
import warnings

import pytest

from docseek.eval.metrics import (
    average_precision_at_k,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ranx import Qrels, Run, evaluate

N_QUERIES = 60
DOCS = [f"d{i}" for i in range(40)]
EMPTY = "__empty__"  # ranx needs at least one doc per query


def _random_data(seed: int) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    rnd = random.Random(seed)
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for q in range(N_QUERIES):
        qid = f"q{q:02d}"
        qrels[qid] = {d: rnd.choice([1, 2]) for d in rnd.sample(DOCS, rnd.randint(1, 8))}
        retrieved = rnd.sample(DOCS, rnd.randint(0, 25))
        scores = rnd.sample(range(1, 1000), len(retrieved))  # distinct: no tie ambiguity
        run[qid] = {d: float(s) for d, s in zip(retrieved, scores, strict=True)} or {EMPTY: 0.0}
    return qrels, run


def _ranked(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda d: -scores[d])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("k", [1, 3, 5, 10, 20])
def test_metrics_match_ranx(seed: int, k: int) -> None:
    qrels, run = _random_data(seed)
    names = [f"recall@{k}", f"precision@{k}", f"mrr@{k}", f"ndcg_burges@{k}", f"hit_rate@{k}"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = evaluate(Qrels(qrels), Run(run), [*names, f"map@{k}"], return_mean=False)
    ours_fns = [recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k, hit_at_k]
    qids = sorted(qrels)  # ranx reports queries in sorted id order
    for name, fn in zip(names, ours_fns, strict=True):
        for i, qid in enumerate(qids):
            ours = fn(_ranked(run[qid]), qrels[qid], k)
            assert abs(ours - float(theirs[name][i])) < 1e-6, (name, qid)
    for i, qid in enumerate(qids):
        # ranx divides AP by |R|; the spec divides by min(|R|, k). Convert, then compare.
        n_rel = len(qrels[qid])
        expected = float(theirs[f"map@{k}"][i]) * n_rel / min(n_rel, k)
        assert abs(average_precision_at_k(_ranked(run[qid]), qrels[qid], k) - expected) < 1e-6
