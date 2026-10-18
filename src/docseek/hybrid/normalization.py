"""Score normalisation for score-based fusion."""

from __future__ import annotations

import math
from collections.abc import Sequence


def min_max(scores: Sequence[float]) -> list[float]:
    """Scale to [0, 1]; all-equal scores map to 1.0 (every item is the best item)."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def z_score(scores: Sequence[float]) -> list[float]:
    """(s - mean) / population std; all-equal scores map to 0.0."""
    if not scores:
        return []
    n = len(scores)
    mean = sum(scores) / n
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / n)
    if std == 0:
        return [0.0] * n
    return [(s - mean) / std for s in scores]


NORMALIZERS = {"minmax": min_max, "zscore": z_score}
