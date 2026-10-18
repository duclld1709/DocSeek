"""TF-IDF cosine baseline over one field.

``w(t, d) = (1 + ln tf(t, d)) * idf(t)`` with the smoothed idf ``ln((N + 1) / (df + 1)) + 1``
(scikit-learn's default). Documents and the query are L2-normalised, so the score is cosine
similarity. Query term frequencies count, unlike BM25 here.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from docseek.lexical.inverted_index import F32Array, InvertedIndex


def smooth_idf(n_docs: int, df: int) -> float:
    return math.log((n_docs + 1) / (df + 1)) + 1.0


class TfidfScorer:
    def __init__(self, index: InvertedIndex, *, field: str | None = None) -> None:
        self.index = index
        self.field = field or index.fields[0]
        if self.field not in index.fields:
            raise ValueError(f"field {self.field!r} is not indexed")
        n = index.n_docs
        sq = np.zeros(n, dtype=np.float64)
        for term, p in index.postings[self.field].items():
            w = (1 + np.log(p.tfs.astype(np.float64))) * smooth_idf(n, index.df[term])
            np.add.at(sq, p.docs, w * w)
        self._doc_norm = np.sqrt(sq).astype(np.float32)

    def score(self, terms: Sequence[str]) -> F32Array:
        n = self.index.n_docs
        scores = np.zeros(n, dtype=np.float32)
        q_weights: dict[str, float] = {}
        for term, tf in Counter(terms).items():
            df = self.index.df.get(term)
            if df:
                q_weights[term] = (1 + math.log(tf)) * smooth_idf(n, df)
        q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
        if q_norm == 0:
            return scores
        postings = self.index.postings[self.field]
        for term, qw in q_weights.items():
            p = postings.get(term)
            if p is None:
                continue
            dw = (1 + np.log(p.tfs)) * smooth_idf(n, self.index.df[term])
            scores[p.docs] += qw * dw
        nz = self._doc_norm > 0
        scores[nz] /= self._doc_norm[nz] * q_norm
        return scores
