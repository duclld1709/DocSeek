"""Bootstrap CI, paired randomization test, comparison verdicts and weighted kappa."""

import numpy as np
import pytest

from docseek.eval.stats import (
    bootstrap_ci,
    cohen_kappa,
    compare_paired,
    paired_randomization_test,
)

# Hand-computed kappa example: both raters use each label twice.
A = [0, 1, 2, 2, 1, 0]
B = [0, 2, 2, 1, 1, 0]


def test_kappa_unweighted_by_hand() -> None:
    # p_o = 4/6, p_e = 3 * (1/3 * 1/3) = 1/3  ->  (2/3 - 1/3) / (2/3) = 0.5
    assert cohen_kappa(A, B, weights="none") == pytest.approx(0.5)


def test_kappa_linear_by_hand() -> None:
    # D_o = (0.5 + 0.5) / 6 = 1/6; D_e = (1/9) * 4 = 4/9  ->  1 - 9/24 = 0.625
    assert cohen_kappa(A, B, weights="linear") == pytest.approx(0.625)


def test_kappa_quadratic_by_hand() -> None:
    # D_o = (0.25 + 0.25) / 6 = 1/12; D_e = (1/9) * 3 = 1/3  ->  1 - 1/4 = 0.75
    assert cohen_kappa(A, B, weights="quadratic") == pytest.approx(0.75)


def test_kappa_extremes() -> None:
    assert cohen_kappa([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
    assert cohen_kappa([2, 2, 2], [2, 2, 2]) == 1.0
    assert cohen_kappa([0, 2, 0, 2], [2, 0, 2, 0], weights="none") == pytest.approx(-1.0)


def test_kappa_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([0, 1], [0])
    with pytest.raises(ValueError, match="label outside"):
        cohen_kappa([0, 3], [0, 1])


def test_bootstrap_ci_contains_mean_and_is_seeded() -> None:
    values = list(np.random.default_rng(0).random(200))
    ci = bootstrap_ci(values)
    assert ci.low <= ci.mean <= ci.high and ci.n == 200
    assert ci == bootstrap_ci(values)  # same seed -> identical
    assert bootstrap_ci(values, seed=1) != ci
    # roughly mean +/- 1.96 * sd / sqrt(n) for uniform(0, 1)
    assert ci.high - ci.low == pytest.approx(2 * 1.96 * (1 / 12) ** 0.5 / 200**0.5, rel=0.25)


def test_bootstrap_of_constant_is_a_point() -> None:
    ci = bootstrap_ci([0.5] * 10)
    assert (ci.mean, ci.low, ci.high) == (0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_randomization_test() -> None:
    rng = np.random.default_rng(1)
    base = rng.random(100)
    assert paired_randomization_test(base, base) == 1.0
    assert paired_randomization_test(base, base + 0.05) < 0.001  # consistent improvement
    noise = base + rng.normal(0, 0.2, 100)
    assert paired_randomization_test(base, noise - (noise - base).mean()) > 0.5  # zero mean diff
    with pytest.raises(ValueError):
        paired_randomization_test([1.0], [1.0, 2.0])


def test_compare_paired_verdicts() -> None:
    rng = np.random.default_rng(2)
    a = list(rng.random(80))
    better = compare_paired(a, [x + 0.1 for x in a])
    assert better.verdict == "better" and better.delta == pytest.approx(0.1)
    assert better.delta_ci.low == pytest.approx(0.1) and better.p_value < 0.05
    assert compare_paired(a, [x - 0.1 for x in a]).verdict == "worse"
    assert compare_paired(a, a).verdict == "no significant difference"
