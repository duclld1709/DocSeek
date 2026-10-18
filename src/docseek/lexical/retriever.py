"""Lexical retrievers over chunks: own BM25/BM25F/TF-IDF, and ``bm25s`` as a reference.

ACL and metadata filters are applied as a **pre-filter**: unreadable or filtered-out chunks
are masked before top-k selection, so a query always gets up to ``k`` readable hits and
nothing it may not see can ever be ranked.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from docseek.core.acl import AccessPolicy
from docseek.core.filtering import matches_filter
from docseek.core.models import Chunk, MetadataFilter, Query, ScoredHit
from docseek.lexical.inverted_index import F32Array
from docseek.lexical.tokenizer import Tokenizer, TokenizerConfig

BoolArray = npt.NDArray[np.bool_]


class Scorer(Protocol):
    def score(self, terms: Sequence[str]) -> F32Array: ...


class VisibilityMask:
    """Boolean masks over a fixed chunk list: readable by a principal, matching a filter."""

    def __init__(self, chunks: Sequence[Chunk], policy: AccessPolicy) -> None:
        self._chunks = chunks
        self._policy = policy
        self._acl_cache: dict[str, BoolArray] = {}

    def mask(self, principal: str, flt: MetadataFilter | None) -> BoolArray:
        acl = self._acl_cache.get(principal)
        if acl is None:
            groups = self._policy.groups(principal)  # unknown principals raise: fail closed
            acl = np.fromiter(
                (not groups.isdisjoint(c.metadata["acl"]) for c in self._chunks),
                dtype=np.bool_,
                count=len(self._chunks),
            )
            self._acl_cache[principal] = acl
        if flt is None:
            return acl
        keep = np.fromiter(
            (matches_filter(c.metadata, flt) for c in self._chunks),
            dtype=np.bool_,
            count=len(self._chunks),
        )
        return acl & keep


def top_k_hits(
    scores: F32Array,
    mask: BoolArray,
    chunks: Sequence[Chunk],
    k: int,
    retriever: str,
    *,
    debug: dict[str, Any] | None = None,
) -> list[ScoredHit]:
    """Best ``k`` chunks with a positive score among ``mask``; ties broken by chunk order."""
    candidates = np.flatnonzero(mask & (scores > 0))
    if candidates.size == 0:
        return []
    order = candidates[np.argsort(-scores[candidates], kind="stable")][:k]
    return [
        ScoredHit(
            chunk_id=chunks[i].chunk_id,
            section_id=chunks[i].section_id,
            score=float(scores[i]),
            rank=rank,
            retriever=retriever,
            debug=dict(debug) if debug else {},
        )
        for rank, i in enumerate(order, start=1)
    ]


class LexicalRetriever:
    """Tokenize the query, score every chunk, pre-filter by ACL/metadata, take top-k."""

    def __init__(
        self,
        name: str,
        chunks: Sequence[Chunk],
        scorer: Scorer,
        tokenizer: TokenizerConfig,
        policy: AccessPolicy,
    ) -> None:
        self.name = name
        self._chunks = list(chunks)
        self._scorer = scorer
        self._tokenizer = Tokenizer(tokenizer)
        self._visible = VisibilityMask(self._chunks, policy)

    def search(self, query: Query, k: int) -> list[ScoredHit]:
        mask = self._visible.mask(query.principal, query.filters)
        scores = self._scorer.score(self._tokenizer.tokenize(query.text))
        return top_k_hits(scores, mask, self._chunks, k, self.name)


class Bm25sScorer:
    """``bm25s`` (Lucene variant) over the same tokens as our index; reference only."""

    def __init__(
        self, chunks_terms: Sequence[list[str]], *, k1: float = 1.2, b: float = 0.75
    ) -> None:
        import bm25s

        self._model = bm25s.BM25(k1=k1, b=b, method="lucene")
        self._model.index(list(chunks_terms), show_progress=False)
        self._vocab: dict[str, int] = self._model.vocab_dict

    def score(self, terms: Sequence[str]) -> F32Array:
        known = [t for t in dict.fromkeys(terms) if t in self._vocab]
        if not known:
            return np.zeros(self._model.scores["num_docs"], dtype=np.float32)
        return np.asarray(self._model.get_scores(known), dtype=np.float32)
