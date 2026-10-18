"""BM25 and a simple BM25F (per-field length normalisation, then one saturation).

For a query term ``t`` and document ``d``::

    B_f(d)  = 1 - b + b * len_f(d) / avglen_f
    tf~(d)  = sum_f w_f * tf_f(t, d) / B_f(d)
    idf(t)  = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))        # Lucene, never negative
    s(t, d) = idf(t) * tf~(d) * (k1 + 1) / (k1 + tf~(d))

With a single field of weight 1 this is exactly Lucene's BM25:
``idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``. Query terms are de-duplicated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from docseek.lexical.inverted_index import F32Array, InvertedIndex


def lucene_idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


class BM25Scorer:
    def __init__(
        self,
        index: InvertedIndex,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        field_weights: Mapping[str, float] | None = None,
    ) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("need k1 >= 0 and 0 <= b <= 1")
        weights = dict(field_weights) if field_weights else dict.fromkeys(index.fields, 1.0)
        unknown = set(weights) - set(index.fields)
        if unknown:
            raise ValueError(f"field weights for unindexed fields: {sorted(unknown)}")
        self.index, self.k1, self.b = index, k1, b
        self.weights = {f: w for f, w in weights.items() if w > 0}
        # Per-field length normaliser B_f(d), precomputed once.
        self._norm: dict[str, F32Array] = {}
        for f in self.weights:
            avg = index.avg_length(f)
            rel = index.lengths[f] / avg if avg > 0 else np.ones_like(index.lengths[f])
            self._norm[f] = ((1 - b) + b * rel).astype(np.float32)

    def score(self, terms: Sequence[str]) -> F32Array:
        """Scores of every document for the query terms; 0 where no term matches."""
        n = self.index.n_docs
        scores = np.zeros(n, dtype=np.float32)
        for term in dict.fromkeys(terms):
            df = self.index.df.get(term)
            if not df:
                continue
            tf = np.zeros(n, dtype=np.float32)
            for f, w in self.weights.items():
                p = self.index.postings[f].get(term)
                if p is not None:
                    tf[p.docs] += w * p.tfs / self._norm[f][p.docs]
            hit = tf > 0
            idf = lucene_idf(n, df)
            scores[hit] += idf * tf[hit] * (self.k1 + 1) / (self.k1 + tf[hit])
        return scores
