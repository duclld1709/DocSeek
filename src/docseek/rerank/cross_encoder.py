"""Cross-encoder reranking (fastembed ONNX, CPU) and a retriever wrapper that applies it."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

from docseek.core.interfaces import Reranker, Retriever
from docseek.core.models import Chunk, Query, ScoredHit


class ScoreCache:
    """SQLite map sha256(model, query, passage) -> score, so reruns and larger N are cheap."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS s (k TEXT PRIMARY KEY, v REAL)")

    @staticmethod
    def key(model: str, query: str, passage: str) -> str:
        return hashlib.sha256(chr(31).join((model, query, passage)).encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for i in range(0, len(keys), 500):
            part = keys[i : i + 500]
            marks = ",".join("?" * len(part))
            out.update(self._db.execute(f"SELECT k, v FROM s WHERE k IN ({marks})", part))
        return out

    def put_many(self, items: dict[str, float]) -> None:
        self._db.executemany("INSERT OR REPLACE INTO s VALUES (?, ?)", items.items())
        self._db.commit()


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly; higher is more relevant.

    ``max_chars`` truncates passages before tokenisation (CPU cost grows with length).
    """

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 8,  # DirectML slows down sharply with large batches
        cache_path: Path | None = None,
        max_chars: int | None = None,
    ) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # heavy import

        from docseek.core.onnx import onnx_providers

        self.name = f"ce:{model_name.split('/')[-1]}"
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name, providers=list(onnx_providers()))
        self._batch_size = batch_size
        self._cache = ScoreCache(cache_path) if cache_path else None
        self._max_chars = max_chars
        self.scored = 0  # pairs actually run through the model

    def score(self, query: str, docs: list[str]) -> list[float]:
        if self._max_chars:
            docs = [d[: self._max_chars] for d in docs]
        if self._cache is None:
            self.scored += len(docs)
            return [float(x) for x in self._model.rerank(query, docs, batch_size=self._batch_size)]
        keys = [ScoreCache.key(self.model_name, query, d) for d in docs]
        known = self._cache.get_many(keys)
        todo = [i for i, k in enumerate(keys) if k not in known]
        if todo:
            new = self._model.rerank(query, [docs[i] for i in todo], batch_size=self._batch_size)
            fresh = {keys[i]: float(v) for i, v in zip(todo, new, strict=True)}
            self._cache.put_many(fresh)
            known.update(fresh)
            self.scored += len(todo)
        return [known[k] for k in keys]

    def rerank(
        self, query: str, hits: list[ScoredHit], texts: dict[str, str], k: int
    ) -> list[ScoredHit]:
        if not hits:
            return []
        scores = self.score(query, [texts[h.chunk_id] for h in hits])
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], hits[i].rank))[:k]
        return [
            hits[i].model_copy(
                update={
                    "score": float(scores[i]),
                    "rank": r,
                    "retriever": f"{hits[i].retriever}+{self.name}",
                    "debug": {**hits[i].debug, "first_stage_rank": hits[i].rank},
                }
            )
            for r, i in enumerate(order, start=1)
        ]


class RerankedRetriever:
    """First stage to ``top_n``, then rerank to ``k``. ACL is enforced by the first stage."""

    def __init__(
        self,
        name: str,
        first_stage: Retriever,
        reranker: Reranker,
        texts: Mapping[str, str],
        *,
        top_n: int = 50,
    ) -> None:
        self.name = name
        self._first = first_stage
        self._reranker = reranker
        self._texts = dict(texts)
        self._top_n = top_n

    def search(self, query: Query, k: int) -> list[ScoredHit]:
        hits = self._first.search(query, max(self._top_n, k))
        return self._reranker.rerank(query.text, hits[: self._top_n], self._texts, k)


def rerank_texts(chunks: Sequence[Chunk]) -> dict[str, str]:
    """Passage text shown to the reranker: section path, then the chunk text."""
    out: dict[str, str] = {}
    for c in chunks:
        path = str(c.metadata.get("section_path") or "")
        out[c.chunk_id] = f"{path}\n{c.text}" if path else c.text
    return out
