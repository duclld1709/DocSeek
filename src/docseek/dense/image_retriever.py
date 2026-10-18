"""CLIP text-to-image retrieval over the docs' images (M7).

Image chunks (``strategy="image"``, see ``ingest/images.py``) are embedded from their pixels
with the CLIP vision tower and stored in Qdrant under the named vector ``image``; a query is
embedded with the CLIP text tower. ACL is the same Qdrant ``acl`` filter as for text chunks
(the image chunk carries its owning section's ACL), so a restricted section's image is never
returned to a principal who cannot read the section.

Options (experiment YAML)::

    type: clip
    repo_root: data/raw/fastapi     # image_path in chunk metadata is relative to this
    location: http://127.0.0.1:6333
    cache_dir: data/index/emb       # .npy cache of image vectors; null disables
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from docseek.core.acl import AccessPolicy
from docseek.core.interfaces import FloatArray
from docseek.core.models import Chunk, Query, ScoredHit
from docseek.dense.embedders import _normalise
from docseek.dense.qdrant_store import QdrantStore, connect

IMAGE_VECTOR = "image"


class ImageTextEncoder(Protocol):
    name: str
    dim: int

    def embed_images(self, paths: Sequence[Path]) -> FloatArray: ...

    def embed_query(self, text: str) -> FloatArray: ...


class ClipEncoder:
    """fastembed CLIP towers (models from ``configs/models.yaml`` ``image_embedding``)."""

    def __init__(self, image_model: str, text_model: str, *, batch_size: int = 8) -> None:
        from fastembed import ImageEmbedding, TextEmbedding

        from docseek.core.onnx import onnx_providers

        providers = list(onnx_providers())
        self.name = f"clip:{image_model}"
        self._image = ImageEmbedding(image_model, providers=providers)
        self._text = TextEmbedding(text_model, providers=providers)
        self._batch_size = batch_size
        self.dim = int(self._text.embedding_size)

    def embed_images(self, paths: Sequence[Path]) -> FloatArray:
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = list(self._image.embed([str(p) for p in paths], batch_size=self._batch_size))
        return _normalise(np.stack(vecs).astype(np.float32))

    def embed_query(self, text: str) -> FloatArray:
        (vec,) = list(self._text.embed([text]))
        out: FloatArray = _normalise(np.asarray(vec, dtype=np.float32)[None, :])[0]
        return out


def image_digest(chunks: Sequence[Chunk], repo_root: Path, encoder_name: str) -> str:
    h = hashlib.sha256(encoder_name.encode())
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(hashlib.sha256((repo_root / c.metadata["image_path"]).read_bytes()).digest())
        h.update(repr(sorted(c.metadata.items())).encode())
    return h.hexdigest()[:10]


class ImageRetriever:
    def __init__(
        self, name: str, encoder: ImageTextEncoder, store: QdrantStore, policy: AccessPolicy
    ) -> None:
        self.name = name
        self._encoder = encoder
        self._store = store
        self._policy = policy

    def search(self, query: Query, k: int) -> list[ScoredHit]:
        groups = self._policy.groups(query.principal)  # unknown principal raises: fail closed
        rows = self._store.search(
            self._encoder.embed_query(query.text), k, groups=groups, flt=query.filters
        )
        return [
            ScoredHit(chunk_id=cid, section_id=sid, score=score, rank=i, retriever=self.name)
            for i, (cid, sid, score) in enumerate(rows, start=1)
        ]


def build_image_retriever(
    opts: Mapping[str, Any],
    chunks: Sequence[Chunk],
    policy: AccessPolicy,
    *,
    encoder: ImageTextEncoder | None = None,
) -> ImageRetriever:
    from docseek.core.config import Settings, load_models_config

    images = [c for c in chunks if c.strategy == "image"]
    if not images:
        raise ValueError("clip retriever needs image chunks (strategy 'image') in the corpus")
    root = Path(str(opts.get("repo_root", "data/raw/fastapi")))
    if encoder is None:
        cfg = load_models_config().image_embedding
        encoder = ClipEncoder(cfg["image"], cfg["text"])
    digest = image_digest(images, root, encoder.name)
    store = QdrantStore(
        connect(str(opts.get("location") or Settings().qdrant_url)),
        f"images_{digest}",
        encoder.dim,
        vector_name=IMAGE_VECTOR,
    )
    store.ensure_collection()
    if store.count() != len(images):
        cache = opts.get("cache_dir", "data/index/emb")
        npy = Path(cache) / f"clip_{digest}.npy" if cache else None
        if npy is not None and npy.exists():
            vectors = np.load(npy)
        else:
            vectors = encoder.embed_images([root / c.metadata["image_path"] for c in images])
            if npy is not None:
                npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(npy, vectors)
        store.upsert(images, vectors)
    return ImageRetriever(f"clip-{encoder.name.split('/')[-1]}", encoder, store, policy)
