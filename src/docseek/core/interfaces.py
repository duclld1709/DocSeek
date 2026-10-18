"""Structural interfaces every component implements (PROJECT_SPEC §4.3).

Each protocol has a deterministic fake in ``docseek.testing.fakes`` so unit tests and CI
run without network access or API keys.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from docseek.core.models import LLMResponse, Query, ScoredHit

FloatArray = npt.NDArray[np.float32]


@runtime_checkable
class Retriever(Protocol):
    name: str

    def search(self, query: Query, k: int) -> list[ScoredHit]:
        """Return at most ``k`` hits the query's principal is allowed to see, best first."""
        ...


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> FloatArray:
        """Embed passages; returns an array of shape ``(len(texts), dim)``."""
        ...

    def embed_query(self, text: str) -> FloatArray:
        """Embed one query; returns an array of shape ``(dim,)``."""
        ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, hits: list[ScoredHit], texts: dict[str, str], k: int
    ) -> list[ScoredHit]:
        """Reorder ``hits`` using ``texts`` (keyed by chunk_id) and keep the top ``k``."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Run one chat completion."""
        ...
