"""Embedders implementing ``core.interfaces.Embedder``, plus an on-disk ``.npy`` cache.

fastembed does not add instruction prefixes, so :class:`FastEmbedEmbedder` applies the
model-card prefixes from ``configs/models.yaml`` itself (``query`` vs ``passage`` side).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from docseek.core.interfaces import Embedder, FloatArray


class FastEmbedEmbedder:
    """ONNX embeddings via fastembed (CPU, no torch). Vectors are L2-normalised.

    Args:
        model_name: A fastembed-supported model id, from ``configs/models.yaml``.
        batch_size: Texts per ONNX batch.
        query_prefix: Prepended to queries (model card instruction), e.g. ``"search_query: "``.
        passage_prefix: Prepended to passages.
    """

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 8,  # small batches suit DirectML; CPU speed is similar
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        from fastembed import TextEmbedding  # heavy import, only for real runs

        from docseek.core.onnx import onnx_providers

        self.name = f"fastembed:{model_name}"
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model = TextEmbedding(model_name, providers=list(onnx_providers()))
        self._batch_size = batch_size
        self.dim = int(self._model.embedding_size)

    def embed_documents(self, texts: list[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefixed = [self.passage_prefix + t for t in texts]
        vectors = np.stack(list(self._model.embed(prefixed, batch_size=self._batch_size)))
        return _normalise(vectors.astype(np.float32))

    def embed_query(self, text: str) -> FloatArray:
        (vec,) = list(self._model.embed([self.query_prefix + text]))
        out: FloatArray = _normalise(np.asarray(vec, dtype=np.float32)[None, :])[0]
        return out


def _normalise(vectors: FloatArray) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", name)


class _VectorStore:
    """Append-only ``vectors.npy`` + ``keys.json`` in one directory."""

    def __init__(self, path: Path, dim: int) -> None:
        self.path = path
        self.keys: dict[str, int] = {}
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        if (path / "keys.json").is_file():
            self.keys = json.loads((path / "keys.json").read_text(encoding="utf-8"))
            self.vectors = np.load(path / "vectors.npy")

    def add(self, keys: list[str], vectors: FloatArray) -> None:
        start = len(self.vectors)
        self.vectors = np.concatenate([self.vectors, vectors.astype(np.float32)])
        for i, k in enumerate(keys):
            self.keys[k] = start + i
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / "vectors.npy", self.vectors)
        (self.path / "keys.json").write_text(json.dumps(self.keys), encoding="utf-8")


class CachedEmbedder:
    """Wraps an embedder; vectors persist under ``cache_dir/<model>/{passage,query}/``.

    The key of a text is ``sha256(embedder name, side, prefix, text)``, so changing the model
    or a prefix never returns stale vectors. Only missing texts are embedded.
    """

    def __init__(self, inner: Embedder, cache_dir: Path) -> None:
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        root = cache_dir / _slug(inner.name)
        self._stores = {side: _VectorStore(root / side, self.dim) for side in ("passage", "query")}
        self.hits = 0
        self.misses = 0

    def _key(self, side: str, text: str) -> str:
        prefix = getattr(self.inner, f"{side}_prefix", "")
        payload = json.dumps([self.inner.name, side, prefix, text], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _lookup(self, side: str, texts: list[str]) -> FloatArray:
        store = self._stores[side]
        keys = [self._key(side, t) for t in texts]
        missing = list(dict.fromkeys(k for k in keys if k not in store.keys))
        self.hits += len(keys) - len(missing)
        self.misses += len(missing)
        if missing:
            by_key = dict(zip(keys, texts, strict=True))
            todo = [by_key[k] for k in missing]
            new = (
                self.inner.embed_documents(todo)
                if side == "passage"
                else np.stack([self.inner.embed_query(t) for t in todo])
            )
            store.add(missing, new)
        rows: FloatArray = store.vectors[[store.keys[k] for k in keys]]
        return rows

    def embed_documents(self, texts: list[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._lookup("passage", texts)

    def embed_query(self, text: str) -> FloatArray:
        out: FloatArray = self._lookup("query", [text])[0]
        return out
