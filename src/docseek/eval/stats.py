"""Statistics for comparing runs and judges (PROJECT_SPEC §5.4, §5.6).

- Percentile bootstrap CI over queries (1,000 resamples by default).
- Paired randomization (sign-flip) test for the mean per-query difference of two configs.
- Weighted Cohen's kappa for judge-vs-human agreement on the 0/1/2 relevance scale.

Everything is seeded; the seed belongs in ``env.json`` of the run that used it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

DEFAULT_SEED = 13
ALPHA = 0.05


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    low: float
    high: float
    n: int


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    alpha: float = ALPHA,
    seed: int = DEFAULT_SEED,
) -> ConfidenceInterval:
    """Percentile bootstrap CI of the mean, resampling queries with replacement."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("bootstrap needs at least one value")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return ConfidenceInterval(float(arr.mean()), float(low), float(high), int(arr.size))


def paired_randomization_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_permutations: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> float:
    """Two-sided p-value for H0: configs A and B have the same mean per-query score.

    Under H0 the sign of each per-query difference is exchangeable, so signs are flipped at
    random; ``p = (#{|mean(flipped)| >= |mean(observed)|} + 1) / (n_permutations + 1)``.
    """
    da = np.asarray(a, dtype=np.float64)
    db = np.asarray(b, dtype=np.float64)
    if da.shape != db.shape or da.size == 0:
        raise ValueError("need two non-empty score lists of the same length (paired by query)")
    diffs = da - db
    observed = abs(diffs.mean())
    if np.allclose(diffs, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, diffs.size))
    flipped = np.abs((signs * diffs).mean(axis=1))
    count = int(np.sum(flipped >= observed - 1e-12))
    return (count + 1) / (n_permutations + 1)


Verdict = Literal["better", "worse", "no significant difference"]


@dataclass(frozen=True)
class Comparison:
    """B relative to A: ``delta = mean(B) - mean(A)``."""

    mean_a: float
    mean_b: float
    delta: float
    delta_ci: ConfidenceInterval
    p_value: float
    verdict: Verdict


def compare_paired(
    a: Sequence[float],
    b: Sequence[float],
    *,
    alpha: float = ALPHA,
    seed: int = DEFAULT_SEED,
) -> Comparison:
    """Δ, its bootstrap CI and p-value; "better" only when Δ > 0 and p < alpha (§5.6)."""
    p = paired_randomization_test(a, b, seed=seed)
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    ci = bootstrap_ci(diffs, alpha=alpha, seed=seed)
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    delta = mean_b - mean_a
    verdict: Verdict = "no significant difference"
    if p < alpha:
        verdict = "better" if delta > 0 else "worse"
    return Comparison(mean_a, mean_b, delta, ci, p, verdict)


Weights = Literal["none", "linear", "quadratic"]


def cohen_kappa(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    *,
    labels: Sequence[int] = (0, 1, 2),
    weights: Weights = "linear",
) -> float:
    """Cohen's kappa ``1 - observed_disagreement / expected_disagreement``.

    Disagreement weights: ``none`` 0/1, ``linear`` |i-j|/(L-1), ``quadratic`` that squared.
    Returns 1.0 when both raters use a single identical label (no disagreement possible).
    """
    if len(rater_a) != len(rater_b) or not rater_a:
        raise ValueError("need two non-empty label lists of the same length")
    index = {label: i for i, label in enumerate(labels)}
    n_labels = len(labels)
    confusion = np.zeros((n_labels, n_labels), dtype=np.float64)
    for x, y in zip(rater_a, rater_b, strict=True):
        if x not in index or y not in index:
            raise ValueError(f"label outside {list(labels)}: {x!r}, {y!r}")
        confusion[index[x], index[y]] += 1
    i, j = np.indices((n_labels, n_labels))
    if weights == "none":
        w = (i != j).astype(np.float64)
    else:
        w = np.abs(i - j) / max(1, n_labels - 1)
        if weights == "quadratic":
            w = w**2
    observed = confusion / confusion.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    expected_disagreement = float((w * expected).sum())
    if expected_disagreement == 0.0:
        return 1.0
    return 1.0 - float((w * observed).sum()) / expected_disagreement
