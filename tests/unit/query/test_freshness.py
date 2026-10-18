import math
from datetime import UTC, datetime

import pytest

from docseek.core.models import ScoredHit
from docseek.query.freshness import FreshnessConfig, multiplier

REF = datetime(2026, 9, 1, tzinfo=UTC)


def test_multiplier_hand_computed() -> None:
    cfg = FreshnessConfig(beta=0.3, tau_days=180, deprecated_penalty=0.5, superseded_penalty=0.5)
    # Updated on the reference day: 1 + 0.3 * e^0 = 1.3
    assert multiplier(REF, "active", False, REF, cfg) == pytest.approx(1.3)
    # 180 days old: 1 + 0.3 / e
    old = datetime(2026, 3, 5, tzinfo=UTC)  # 180 days before REF
    assert (REF - old).days == 180
    assert multiplier(old, "active", False, REF, cfg) == pytest.approx(1 + 0.3 / math.e)
    # Deprecated and superseded: 1.3 * 0.5 * 0.5
    assert multiplier(REF, "deprecated", True, REF, cfg) == pytest.approx(0.325)
    # No date: no boost; future dates count as age 0.
    assert multiplier(None, "active", False, REF, cfg) == 1.0
    assert multiplier(datetime(2027, 1, 1, tzinfo=UTC), "active", False, REF, cfg) == pytest.approx(
        1.3
    )


def test_rescore_demotes_stale_version() -> None:
    from docseek.core.models import Chunk
    from docseek.query.freshness import FreshnessReranker

    def chunk(cid: str, doc: str, date: str, status: str, sup: str | None) -> Chunk:
        return Chunk(
            chunk_id=f"{cid}::section::0",
            section_id=cid,
            strategy="section",
            text="t",
            text_for_embedding="t",
            metadata={"doc_id": doc, "updated_at": date, "status": status, "supersedes": sup},
            n_tokens=1,
        )

    old = chunk("s:old.md#a", "s:old.md", "2026-01-01", "deprecated", None)
    new = chunk("s:new.md#a", "s:new.md", "2026-08-01", "active", "s:old.md")
    r = FreshnessReranker("f", inner=None, chunks=[old, new], cfg=FreshnessConfig())  # type: ignore[arg-type]
    hits = [
        ScoredHit(
            chunk_id=old.chunk_id, section_id=old.section_id, score=10, rank=1, retriever="x"
        ),
        ScoredHit(chunk_id=new.chunk_id, section_id=new.section_id, score=6, rank=2, retriever="x"),
    ]
    out = r.rescore(hits, 2)
    assert [h.section_id for h in out] == ["s:new.md#a", "s:old.md#a"]
    assert out[0].debug["pre_freshness_rank"] == 2


def test_sigmoid_transform_keeps_penalty_direction_for_negative_scores() -> None:
    from docseek.core.models import Chunk
    from docseek.query.freshness import FreshnessReranker

    def chunk(cid: str, doc: str, sup: str | None) -> Chunk:
        return Chunk(
            chunk_id=f"{cid}::section::0",
            section_id=cid,
            strategy="section",
            text="t",
            text_for_embedding="t",
            metadata={"doc_id": doc, "updated_at": None, "status": "active", "supersedes": sup},
            n_tokens=1,
        )

    old, new = chunk("s:old.md#a", "s:old.md", None), chunk("s:new.md#a", "s:new.md", "s:old.md")
    hits = [
        ScoredHit(
            chunk_id=old.chunk_id, section_id=old.section_id, score=-1.0, rank=1, retriever="x"
        ),
        ScoredHit(
            chunk_id=new.chunk_id, section_id=new.section_id, score=-1.5, rank=2, retriever="x"
        ),
    ]
    raw = FreshnessReranker("f", None, [old, new], FreshnessConfig(beta=0))  # type: ignore[arg-type]
    sig = FreshnessReranker(
        "f", None, [old, new], FreshnessConfig(beta=0), score_transform="sigmoid"
    )  # type: ignore[arg-type]
    # Raw negative logits: halving -1.0 to -0.5 would *promote* the superseded doc.
    assert raw.rescore(hits, 2)[0].section_id == "s:old.md#a"
    # sigmoid(-1)*0.5 = 0.134 < sigmoid(-1.5) = 0.182: the superseded doc drops.
    assert sig.rescore(hits, 2)[0].section_id == "s:new.md#a"
