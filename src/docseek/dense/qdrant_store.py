"""Qdrant collection for chunks: named vectors, payload indexes, idempotent upserts.

- Point id = UUID5(chunk_id): re-upserting the same chunk overwrites it, never duplicates.
- Payload = the chunk's flat metadata + ``chunk_id``/``section_id``.
- ACL is part of every query filter (``acl`` must share a group with the principal); a
  :class:`MetadataFilter` is translated into additional ``must`` conditions.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from docseek.core.interfaces import FloatArray
from docseek.core.models import Chunk, MetadataFilter

DENSE = "dense"
POINT_NAMESPACE = uuid.UUID("6f1c5c4e-8a7b-4d7e-9a51-3d2f0c1b9e10")
KEYWORD_FIELDS = ("acl", "doc_type", "source", "status", "modality", "section_id")
DATETIME_FIELDS = ("updated_at",)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def connect(location: str) -> QdrantClient:
    """``":memory:"`` (tests), an ``http(s)://`` URL (docker) or a local directory path."""
    if location == ":memory:":
        return QdrantClient(":memory:")
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location, timeout=60)
    return QdrantClient(path=location)


def to_qdrant_filter(groups: Iterable[str], flt: MetadataFilter | None) -> models.Filter:
    """ACL condition (always) plus the metadata filter's conditions."""
    must: list[models.Condition] = [
        models.FieldCondition(key="acl", match=models.MatchAny(any=sorted(groups)))
    ]
    if flt is not None:
        for key in ("doc_type", "source", "modality", "status"):
            values = getattr(flt, key)
            if values is not None:
                must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(values))))
        if flt.version is not None:
            must.append(
                models.FieldCondition(key="version", match=models.MatchValue(value=flt.version))
            )
        if flt.updated_after is not None or flt.updated_before is not None:
            must.append(
                models.FieldCondition(
                    key="updated_at",
                    range=models.DatetimeRange(gte=flt.updated_after, lte=flt.updated_before),
                )
            )
    return models.Filter(must=must)


class QdrantStore:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dim: int,
        *,
        distance: models.Distance = models.Distance.COSINE,
        on_disk: bool = False,
        quantization: models.QuantizationConfig | None = None,
        hnsw_ef_construct: int | None = None,
        vector_name: str = DENSE,
    ) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim
        self.distance = distance
        self._on_disk = on_disk
        self._quantization = quantization
        self._hnsw_ef_construct = hnsw_ef_construct
        self.vector_name = vector_name

    def ensure_collection(self) -> bool:
        """Create the collection and payload indexes if missing; True if it was created."""
        if self.client.collection_exists(self.collection):
            return False
        hnsw = (
            models.HnswConfigDiff(ef_construct=self._hnsw_ef_construct)
            if self._hnsw_ef_construct
            else None
        )
        self.client.create_collection(
            self.collection,
            vectors_config={
                self.vector_name: models.VectorParams(
                    size=self.dim, distance=self.distance, on_disk=self._on_disk
                )
            },
            quantization_config=self._quantization,
            hnsw_config=hnsw,
        )
        schemas = {n: models.PayloadSchemaType.KEYWORD for n in KEYWORD_FIELDS}
        schemas.update({n: models.PayloadSchemaType.DATETIME for n in DATETIME_FIELDS})
        with warnings.catch_warnings():
            # Local mode (":memory:"/path) ignores payload indexes and warns; server uses them.
            warnings.simplefilter("ignore", UserWarning)
            for name, schema in schemas.items():
                self.client.create_payload_index(self.collection, name, field_schema=schema)
        return True

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def upsert(self, chunks: Sequence[Chunk], vectors: FloatArray, *, batch: int = 256) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        for start in range(0, len(chunks), batch):
            part = zip(chunks[start : start + batch], vectors[start : start + batch], strict=True)
            points = [
                models.PointStruct(
                    id=point_id(c.chunk_id),
                    vector={self.vector_name: v.tolist()},
                    payload={**c.metadata, "chunk_id": c.chunk_id, "section_id": c.section_id},
                )
                for c, v in part
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    def search(
        self,
        vector: FloatArray,
        k: int,
        *,
        groups: Iterable[str],
        flt: MetadataFilter | None = None,
        hnsw_ef: int | None = None,
        exact: bool = False,
    ) -> list[tuple[str, str, float]]:
        """``(chunk_id, section_id, score)`` best first, restricted to readable points."""
        params = models.SearchParams(hnsw_ef=hnsw_ef, exact=exact) if (hnsw_ef or exact) else None
        res = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            using=self.vector_name,
            query_filter=to_qdrant_filter(groups, flt),
            limit=k,
            with_payload=["chunk_id", "section_id"],
            search_params=params,
        )
        out: list[tuple[str, str, float]] = []
        for p in res.points:
            payload: dict[str, Any] = p.payload or {}
            out.append((str(payload["chunk_id"]), str(payload["section_id"]), float(p.score)))
        return out
