"""Embedders implementing ``core.interfaces.Embedder``.

M1 only needs embeddings to de-duplicate synthetic queries and to find hard-negative
candidates, so this holds the minimal fastembed wrapper. M3 adds batching options, the
on-disk ``.npy`` cache and sentence-transformers for model comparisons.
"""

from __future__ import annotations

import numpy as np

from docseek.core.interfaces import FloatArray


class FastEmbedEmbedder:
    """ONNX embeddings via fastembed (CPU, no torch). Vectors are L2-normalised.

    Args:
        model_name: A fastembed-supported model id, from ``configs/models.yaml``.
        batch_size: Texts per ONNX batch.
    """

    def __init__(self, model_name: str, *, batch_size: int = 64) -> None:
        from fastembed import TextEmbedding  # heavy import, only for real runs

        self.name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name)
        self._batch_size = batch_size
        self.dim = int(self._model.embedding_size)

    def embed_documents(self, texts: list[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = np.stack(list(self._model.embed(texts, batch_size=self._batch_size)))
        return _normalise(vectors.astype(np.float32))

    def embed_query(self, text: str) -> FloatArray:
        (vec,) = list(self._model.query_embed(text))
        out: FloatArray = _normalise(np.asarray(vec, dtype=np.float32)[None, :])[0]
        return out


def _normalise(vectors: FloatArray) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)
