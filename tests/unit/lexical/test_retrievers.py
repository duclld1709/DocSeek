"""Lexical retrievers on the fixture corpora: ACL pre-filter (Leakage = 0), filters, ranking."""

import itertools
from datetime import datetime
from pathlib import Path

import pytest

from docseek.core.acl import AccessPolicy
from docseek.core.config import UnknownPrincipalError, load_principals
from docseek.core.filtering import matches_filter
from docseek.core.models import MetadataFilter, Query
from docseek.lexical import factory
from docseek.lexical.inverted_index import chunk_fields, load_or_build
from docseek.lexical.tokenizer import TokenizerConfig
from docseek.testing.acl_checks import probe_leaks
from docseek.testing.fixtures import REPO_ROOT, mini_ingest

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))
CHUNKS = mini_ingest().chunks
CHUNK_ACL = {c.chunk_id: c.metadata["acl"] for c in CHUNKS}
FIELDS = {"title": 2.0, "section_path": 1.0, "body": 1.0, "code": 0.5}

BUILDERS = {
    "bm25": lambda: factory.build_bm25({"index_dir": None}, CHUNKS, POLICY),
    "bm25f": lambda: factory.build_bm25({"index_dir": None, "fields": FIELDS}, CHUNKS, POLICY),
    "tfidf": lambda: factory.build_tfidf({"index_dir": None}, CHUNKS, POLICY),
    "bm25s": lambda: factory.build_bm25s({}, CHUNKS, POLICY),
}


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_no_leakage_for_any_principal(kind: str) -> None:
    result = probe_leaks(BUILDERS[kind](), CHUNK_ACL, POLICY)
    assert result.leaks == []
    assert result.restricted_seen["anonymous"] == 0
    assert result.restricted_seen["engineer"] > 0
    assert result.restricted_seen["security"] > result.restricted_seen["engineer"] - 1


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_symbol_query_finds_its_section(kind: str) -> None:
    hits = BUILDERS[kind]().search(Query(qid="x", text="BackgroundTasks"), 3)
    assert hits[0].section_id.startswith("fastapi-docs:reference/background.md#")
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(a.score >= b.score for a, b in itertools.pairwise(hits))


def test_unknown_principal_fails_closed() -> None:
    with pytest.raises(UnknownPrincipalError):
        BUILDERS["bm25"]().search(Query(qid="x", text="payment", principal="intruder"), 5)


def test_metadata_filter_is_a_pre_filter() -> None:
    r = BUILDERS["bm25"]()
    q = Query(
        qid="x",
        text="Payment API timeouts",
        principal="engineer",
        filters=MetadataFilter(status=["active"]),
    )
    hits = r.search(q, 10)
    assert hits and all("kb-0002" not in h.section_id for h in hits)  # deprecated version


def test_matches_filter_dates_and_versions() -> None:
    meta = {"doc_type": "runbook", "status": "active", "version": 2, "updated_at": "2026-08-02"}
    assert matches_filter(meta, None)
    assert matches_filter(meta, MetadataFilter(updated_after=datetime(2026, 8, 1)))
    assert not matches_filter(meta, MetadataFilter(updated_before=datetime(2026, 1, 1)))
    assert not matches_filter(meta, MetadataFilter(version=1))
    assert not matches_filter(
        {"status": "active"}, MetadataFilter(updated_after=datetime(2020, 1, 1))
    )
    assert not matches_filter(meta, MetadataFilter(doc_type=["adr"]))


def test_fields_separate_code_from_prose() -> None:
    chunk = next(c for c in CHUNKS if "```" in c.text)
    f = chunk_fields(chunk)
    assert "```" not in f["body"] and f["code"]
    assert f["title"] == chunk.metadata["title"]


def test_index_cache_is_reused(tmp_path: Path) -> None:
    tok = TokenizerConfig()
    a = load_or_build(CHUNKS, fields=("all",), tokenizer=tok, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.pkl"))) == 1
    b = load_or_build(CHUNKS, fields=("all",), tokenizer=tok, cache_dir=tmp_path)
    assert a.df == b.df
