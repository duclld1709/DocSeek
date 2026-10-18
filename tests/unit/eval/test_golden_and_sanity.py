"""Golden schema/validation/splits and the Random/Oracle sanity retrievers."""

from collections import Counter
from pathlib import Path

import pytest

from docseek.core.acl import AccessPolicy
from docseek.core.config import UnknownPrincipalError, load_principals
from docseek.core.models import QueryType
from docseek.eval.golden import (
    GoldenQuery,
    GoldenValidationError,
    assign_splits,
    load_golden,
    select,
    validate_against_corpus,
    write_golden,
)
from docseek.eval.metrics import hits_to_sections, leakage_at_k, recall_at_k
from docseek.eval.sanity import OracleRetriever, RandomRetriever
from docseek.testing.fixtures import MINI_GOLDEN, REPO_ROOT, mini_ingest

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))
GOLDEN = load_golden(MINI_GOLDEN)
RESULT = mini_ingest()
SECTION_ACL = {s.section_id: s.acl for s in RESULT.sections}
SEC = "fastapi-docs:tutorial/path-params.md#path-parameters"


def _q(**kw: object) -> GoldenQuery:
    base: dict = {"qid": "q0001", "text": "x", "qtype": "factoid", "origin": "synthetic"}
    base.update(kw)
    base.setdefault("relevant", {SEC: 2})
    return GoldenQuery.model_validate(base)


def test_mini_golden_is_valid_against_the_mini_corpus() -> None:
    validate_against_corpus(GOLDEN, SECTION_ACL, POLICY)
    assert len(GOLDEN) == 14


@pytest.mark.parametrize(
    ("kw", "error"),
    [
        ({"qid": "42"}, "qid must look like"),
        ({"relevant": {SEC: 3}}, "grades must be 1 or 2"),
        ({"relevant": {}}, "only unanswerable or permission"),
        ({"qtype": "unanswerable"}, "unanswerable queries have no relevant"),
        ({"hard_negatives": [SEC]}, "both relevant and hard negative"),
        ({"style": "poetic"}, "style"),
        ({"extra_field": 1}, "extra"),
    ],
)
def test_invalid_golden_queries(kw: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _q(**kw)


def test_empty_relevant_allowed_for_unanswerable_and_permission() -> None:
    assert _q(qtype="unanswerable", relevant={}).relevant == {}
    assert _q(qtype="permission", relevant={}, principal="engineer").relevant == {}


def test_validation_rejects_unknown_sections_duplicates_and_hidden_relevant() -> None:
    with pytest.raises(GoldenValidationError, match="unknown sections"):
        validate_against_corpus([_q(relevant={"fastapi-docs:nope.md#x": 2})], SECTION_ACL, POLICY)
    with pytest.raises(GoldenValidationError, match="duplicate"):
        validate_against_corpus([_q(), _q()], SECTION_ACL, POLICY)
    secret = "internal-kb:kb-0004.md#jwt-signing-key-rotation"
    with pytest.raises(GoldenValidationError, match="cannot read"):
        validate_against_corpus(
            [_q(relevant={secret: 2}, principal="engineer")], SECTION_ACL, POLICY
        )
    with pytest.raises(UnknownPrincipalError):
        validate_against_corpus([_q(principal="root")], SECTION_ACL, POLICY)


def test_splits_are_stratified_deterministic_and_about_40_percent_dev() -> None:
    queries = [
        _q(qid=f"q{i:04d}", qtype=qt)
        for i, qt in enumerate(["factoid"] * 10 + ["howto"] * 20 + ["symbol"] * 5)
    ]
    split = assign_splits(queries)
    assert split == assign_splits(list(reversed(queries)))[::-1]
    dev = Counter(q.qtype for q in split if q.split == "dev")
    assert dev == {QueryType.FACTOID: 4, QueryType.HOWTO: 8, QueryType.SYMBOL: 2}
    assert [q.qid for q in split] == [q.qid for q in queries]  # input order kept


def test_golden_round_trip_and_select(tmp_path: Path) -> None:
    path = tmp_path / "golden_v1.jsonl"
    queries = [_q(qid="q0002", reviewed=True, split="test"), _q(qid="q0001")]
    write_golden(path, queries)
    loaded = load_golden(path)
    assert [q.qid for q in loaded] == ["q0001", "q0002"]
    assert [q.qid for q in select(loaded, split="test")] == ["q0002"]
    assert [q.qid for q in select(loaded, reviewed_only=True)] == ["q0002"]


def test_oracle_recall_is_one_and_never_leaks() -> None:
    oracle = OracleRetriever({q.qid: q.relevant for q in GOLDEN}, RESULT.chunks, POLICY)
    for q in GOLDEN:
        ranked = hits_to_sections(oracle.search(q.to_query(), 10))
        groups = POLICY.groups(q.principal)
        assert leakage_at_k(ranked, SECTION_ACL, groups, 10) == 0
        if q.relevant:
            assert recall_at_k(ranked, q.relevant, 10) == 1.0
            assert q.relevant[ranked[0]] == max(q.relevant.values())  # grade 2 first
        else:
            assert ranked == []


def test_random_retriever_respects_acl_and_is_deterministic() -> None:
    rnd = RandomRetriever(RESULT.chunks, POLICY, seed=7)
    q = next(q for q in GOLDEN if q.principal == "anonymous").to_query()
    first = rnd.search(q, 10)
    assert first == rnd.search(q, 10) and len(first) == 10
    assert [h.rank for h in first] == list(range(1, 11))
    for principal in ("anonymous", "engineer", "security"):
        hits = rnd.search(q.model_copy(update={"principal": principal}), 1000)
        ranked = hits_to_sections(hits)
        assert leakage_at_k(ranked, SECTION_ACL, POLICY.groups(principal), 1000) == 0
    n_public = sum(1 for c in RESULT.chunks if "public" in c.metadata["acl"])
    assert len(rnd.search(q, 1000)) == n_public
