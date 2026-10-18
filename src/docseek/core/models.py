"""Pydantic v2 data models shared by every layer (PROJECT_SPEC §4.2)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from docseek.core.ids import make_section_id

Modality = Literal["text", "scan", "image"]
TraceKind = Literal["understand", "retrieve", "fuse", "rerank", "tool_call", "llm_call", "answer"]


class QueryType(StrEnum):
    """Query intent classes used to slice eval results and to route queries."""

    FACTOID = "factoid"
    HOWTO = "howto"
    CONCEPTUAL = "conceptual"
    SYMBOL = "symbol"
    MULTIHOP = "multihop"
    FRESHNESS = "freshness"
    PERMISSION = "permission"
    SCAN = "scan"
    IMAGE = "image"
    UNANSWERABLE = "unanswerable"


class DocSeekModel(BaseModel):
    """Base model: unknown fields are rejected so typos in data files fail loudly."""

    model_config = ConfigDict(extra="forbid")


class Provenance(DocSeekModel):
    """Where a piece of text came from, precise enough to cite it."""

    file: str
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    page: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def _check_line_range(self) -> Self:
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end must be >= line_start")
        return self


class MetadataFilter(DocSeekModel):
    """Metadata constraints extracted from a query.

    ACL is deliberately absent: access control is derived from the principal and enforced
    by retrievers, so a (possibly LLM-generated) filter can never widen permissions.
    """

    doc_type: list[str] | None = None
    source: list[str] | None = None
    modality: list[Modality] | None = None
    status: list[str] | None = None
    updated_after: datetime | None = None
    updated_before: datetime | None = None
    version: int | None = None


class Section(DocSeekModel):
    """A heading-delimited section: the unit relevance is judged at."""

    section_id: str
    doc_id: str
    source: str
    title: str
    anchor: str
    section_path: list[str]
    text: str
    code_refs: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    doc_type: str
    modality: Modality = "text"
    acl: list[str] = Field(min_length=1)
    updated_at: datetime | None
    version: int | None = None
    status: str = "active"
    supersedes: str | None = None
    provenance: Provenance

    @model_validator(mode="after")
    def _check_ids(self) -> Self:
        if self.section_id != make_section_id(self.doc_id, self.anchor):
            raise ValueError(
                f"section_id {self.section_id!r} does not match doc_id + anchor "
                f"({self.doc_id!r}, {self.anchor!r})"
            )
        if not self.doc_id.startswith(f"{self.source}:"):
            raise ValueError(f"doc_id {self.doc_id!r} does not belong to source {self.source!r}")
        return self


class Chunk(DocSeekModel):
    """The unit that gets indexed; depends on the chunking strategy."""

    chunk_id: str
    section_id: str
    strategy: str
    text: str
    text_for_embedding: str
    metadata: dict[str, Any]
    n_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_ids(self) -> Self:
        prefix = f"{self.section_id}::{self.strategy}::"
        if not self.chunk_id.startswith(prefix) or not self.chunk_id[len(prefix) :].isdigit():
            raise ValueError(f"chunk_id {self.chunk_id!r} does not match {prefix!r}<n>")
        return self


class Query(DocSeekModel):
    """A search request. ``principal`` decides which ACL groups may be returned."""

    qid: str
    text: str
    principal: str = "anonymous"
    filters: MetadataFilter | None = None
    qtype: QueryType | None = None


class ScoredHit(DocSeekModel):
    """One ranked result from a retriever, fusion step or reranker."""

    chunk_id: str
    section_id: str
    score: float
    rank: int = Field(ge=1)
    retriever: str
    debug: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(DocSeekModel):
    query: Query
    hits: list[ScoredHit]
    latency_ms: float = Field(ge=0)
    trace_id: str


class ToolCall(DocSeekModel):
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(DocSeekModel):
    """Provider-agnostic result of one LLM completion."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    parsed: dict[str, Any] | None = None
    model: str
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    cached: bool = False


class TraceEvent(DocSeekModel):
    """One step of a request, written to traces.jsonl."""

    trace_id: str
    step: int = Field(ge=0)
    kind: TraceKind
    name: str
    input: dict[str, Any]
    output: dict[str, Any]
    latency_ms: float = Field(ge=0)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
