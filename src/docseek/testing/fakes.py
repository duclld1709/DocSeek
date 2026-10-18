"""Fake implementations of every core interface.

All fakes are deterministic across processes and platforms, so unit tests and CI never
need network access, model downloads or API keys.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel

from docseek.core.interfaces import FloatArray
from docseek.core.models import LLMResponse, ScoredHit


class HashEmbedder:
    """Bag of hashed character n-grams, L2-normalised.

    Texts sharing many n-grams get a high cosine similarity, which is enough to exercise
    vector stores and ranking code. Uses blake2b rather than ``hash()`` because the latter
    is salted per process.
    """

    def __init__(self, dim: int = 64, ngram_range: tuple[int, int] = (3, 5)) -> None:
        lo, hi = ngram_range
        if dim <= 0 or lo <= 0 or hi < lo:
            raise ValueError("dim must be > 0 and ngram_range must satisfy 0 < lo <= hi")
        self.dim = dim
        self.name = f"hash-{dim}"
        self._ngram_range = ngram_range

    def _ngrams(self, text: str) -> Iterable[str]:
        padded = f" {' '.join(text.lower().split())} "
        lo, hi = self._ngram_range
        for n in range(lo, hi + 1):
            for i in range(len(padded) - n + 1):
                yield padded[i : i + n]

    def _embed(self, text: str) -> FloatArray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for gram in self._ngrams(text):
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            sign = 1.0 if value & 1 else -1.0
            vec[(value >> 1) % self.dim] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed_documents(self, texts: list[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed(t) for t in texts])

    def embed_query(self, text: str) -> FloatArray:
        return self._embed(text)


class IdentityReranker:
    """Keeps the incoming order and truncates to ``k``: a no-op baseline for rerank tests."""

    name = "identity"

    def rerank(
        self, query: str, hits: list[ScoredHit], texts: dict[str, str], k: int
    ) -> list[ScoredHit]:
        return [
            hit.model_copy(update={"rank": i, "retriever": f"{hit.retriever}+{self.name}"})
            for i, hit in enumerate(hits[:k], start=1)
        ]


class ScriptExhaustedError(RuntimeError):
    """Raised when a ScriptedLLM is called more times than it has scripted replies."""


@dataclass
class ScriptedLLM:
    """Replays a fixed list of replies, one per ``complete`` call, and records each call.

    A reply may be a plain string (becomes ``content``) or a full ``LLMResponse``. When the
    caller passes ``response_format``, the reply content is validated against it and stored
    in ``parsed``, mirroring structured output from a real provider.
    """

    replies: list[str | LLMResponse]
    model: str = "scripted"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "tools": tools,
                "response_format": response_format,
                "temperature": temperature,
            }
        )
        if len(self.calls) > len(self.replies):
            raise ScriptExhaustedError(
                f"ScriptedLLM has {len(self.replies)} replies but got call #{len(self.calls)}"
            )
        reply = self.replies[len(self.calls) - 1]
        response = (
            reply
            if isinstance(reply, LLMResponse)
            else LLMResponse(content=reply, model=model or self.model)
        )
        if response_format is not None and response.parsed is None:
            if response.content is None:
                raise ValueError("response_format requested but scripted reply has no content")
            parsed = response_format.model_validate(json.loads(response.content))
            response = response.model_copy(update={"parsed": parsed.model_dump(mode="json")})
        return response
