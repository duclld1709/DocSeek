from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from docseek.core.models import (
    Chunk,
    LLMResponse,
    MetadataFilter,
    Provenance,
    Query,
    QueryType,
    ScoredHit,
    Section,
    TraceEvent,
)

DOC_ID = "fastapi-docs:tutorial/first-steps.md"


def _section(**overrides: Any) -> Section:
    fields: dict[str, Any] = {
        "section_id": f"{DOC_ID}#first-steps",
        "doc_id": DOC_ID,
        "source": "fastapi-docs",
        "title": "First Steps",
        "anchor": "first-steps",
        "section_path": ["Tutorial", "First Steps"],
        "text": "Create a file main.py",
        "doc_type": "tutorial",
        "acl": ["public"],
        "updated_at": datetime(2026, 10, 17, tzinfo=UTC),
        "provenance": {"file": "docs/en/docs/tutorial/first-steps.md", "line_start": 1},
    }
    fields.update(overrides)
    return Section.model_validate(fields)


def test_section_defaults_and_json_round_trip() -> None:
    section = _section()

    assert section.modality == "text"
    assert section.status == "active"
    assert section.code_refs == []
    assert Section.model_validate_json(section.model_dump_json()) == section


def test_section_list_defaults_are_not_shared() -> None:
    a, b = _section(), _section()
    a.code_refs.append("docs_src/x.py")
    assert b.code_refs == []


def test_section_rejects_inconsistent_ids() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        _section(section_id=f"{DOC_ID}#other")
    with pytest.raises(ValidationError, match="does not belong"):
        _section(source="internal-kb")


def test_section_requires_non_empty_acl() -> None:
    with pytest.raises(ValidationError):
        _section(acl=[])


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _section(colour="blue")


def test_provenance_line_range() -> None:
    with pytest.raises(ValidationError):
        Provenance(file="a.md", line_start=10, line_end=3)
    assert Provenance(file="a.pdf", page=2, bbox=(0, 0, 10, 10)).page == 2


def test_chunk_id_must_match_section_and_strategy() -> None:
    section_id = f"{DOC_ID}#first-steps"
    ok = Chunk(
        chunk_id=f"{section_id}::section::0",
        section_id=section_id,
        strategy="section",
        text="t",
        text_for_embedding="t",
        metadata={},
        n_tokens=1,
    )
    assert ok.n_tokens == 1
    with pytest.raises(ValidationError):
        ok.model_validate({**ok.model_dump(), "chunk_id": f"{section_id}::fixed_256::0"})
    with pytest.raises(ValidationError):
        ok.model_validate({**ok.model_dump(), "chunk_id": f"{section_id}::section::x"})


def test_query_parses_qtype_and_filters() -> None:
    query = Query.model_validate(
        {
            "qid": "q1",
            "text": "latest key rotation",
            "qtype": "freshness",
            "filters": {"source": ["internal-kb"], "updated_after": "2026-01-01T00:00:00Z"},
        }
    )
    assert query.principal == "anonymous"
    assert query.qtype is QueryType.FRESHNESS
    assert query.filters is not None and query.filters.source == ["internal-kb"]


def test_metadata_filter_cannot_carry_acl() -> None:
    with pytest.raises(ValidationError):
        MetadataFilter.model_validate({"acl": ["security-team"]})


def test_scored_hit_rank_is_one_based() -> None:
    with pytest.raises(ValidationError):
        ScoredHit(chunk_id="c", section_id="s", score=1.0, rank=0, retriever="bm25")


def test_trace_event_kind_is_constrained() -> None:
    with pytest.raises(ValidationError):
        TraceEvent(
            trace_id="t",
            step=0,
            kind="dance",  # type: ignore[arg-type]
            name="x",
            input={},
            output={},
            latency_ms=1.0,
        )


def test_llm_response_defaults() -> None:
    response = LLMResponse(model="fake")
    assert response.tool_calls == []
    assert response.cost_usd == 0.0
    assert response.cached is False
