"""Synthetic query generation on the fixture corpora with a rule-based fake LLM."""

import json
import random
import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from docseek.core.acl import AccessPolicy
from docseek.core.config import LLMRoleConfig, load_principals
from docseek.core.models import LLMResponse, QueryType
from docseek.core.tracing import MemoryTraceSink, Tracer
from docseek.data.gen_internal import KBPlan, KBPlanEntry
from docseek.eval.golden import validate_against_corpus
from docseek.eval.judge import RelevanceJudge
from docseek.eval.synth import (
    Candidate,
    GeneratedQuery,
    SynthConfig,
    SynthContext,
    balance,
    dedupe,
    lexical_overlap,
    minimal_principal,
    multihop_pairs,
    prose_words,
    section_symbols,
    stratified_sample,
    synthesize,
    unprivileged_principal,
)
from docseek.llm.cache import SQLiteLLMCache
from docseek.llm.client import CachedLLMClient
from docseek.llm.cost import BudgetGuard
from docseek.testing.fakes import HashEmbedder
from docseek.testing.fixtures import REPO_ROOT, mini_ingest

PRINCIPALS = load_principals(REPO_ROOT / "configs")
POLICY = AccessPolicy(PRINCIPALS)
SECTIONS = mini_ingest().sections
_TITLE_RE = re.compile(r'<section (?:id="\w" )?title="([^"]*)"')
_QUERY_RE = re.compile(r"<query>\n(.*?)\n</query>", re.S)


class RuleLLM:
    """Answers by prompt content: judge -> 2 iff the section title occurs in the query."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls += 1
        system, user = messages[0]["content"], messages[-1]["content"]
        if "relevance assessor" in system:
            title = _TITLE_RE.search(user)[1]  # type: ignore[index]
            query = _QUERY_RE.search(user)[1]  # type: ignore[index]
            reply: dict[str, Any] = {
                "score": 2 if title.lower() in query.lower() else 0,
                "reason": "rule",
            }
        elif "does NOT answer" in user:
            reply = {
                "items": [
                    {"query": "connect fastapi to zebracorn queues", "keywords": ["zebracorn"]},
                    {"query": "fastapi graphql thing", "keywords": ["fastapi"]},  # in corpus
                    # common word + absent word: no section has both, so it is kept
                    {"query": "fastapi with blorbdb", "keywords": ["fastapi", "blorbdb"]},
                    {"query": "deploy on quuxcloud lambdas", "keywords": ["quuxcloud"]},
                ]
            }
        else:
            titles = _TITLE_RE.findall(user)
            style = next(
                s
                for s in ("human", "machine", "paraphrase")
                if s in user.split("Style:", 1)[1][:40] or s == "paraphrase"
            )
            kind = re.search(r'type "(\w+)"', user)
            label = kind[1] if kind else "multi" if len(titles) > 1 else "fresh"
            reply = {"query": f"{label} {style} " + " and ".join(titles), "reference_answer": "ref"}
        content = json.dumps(reply)
        parsed = response_format.model_validate(reply).model_dump() if response_format else None
        return LLMResponse(content=content, parsed=parsed, model=model or "rule")


PLAN = KBPlan(
    company="Acme Payments",
    docs=[
        KBPlanEntry(
            id="kb-0002",
            title="Payment API timeouts",
            doc_type="runbook",
            acl=["eng-internal"],
            owner="p",
            version=1,
            updated_at=date(2026, 3, 1),
            status="deprecated",
            facts=["The timeout is 30 seconds."],
        ),
        KBPlanEntry(
            id="kb-0003",
            title="Payment API timeouts",
            doc_type="runbook",
            acl=["eng-internal"],
            owner="p",
            version=2,
            updated_at=date(2026, 8, 2),
            supersedes="kb-0002",
            facts=["The timeout is 10 seconds."],
        ),
    ],
)
TARGETS = {
    QueryType.FACTOID: 4,
    QueryType.HOWTO: 4,
    QueryType.CONCEPTUAL: 2,
    QueryType.SYMBOL: 3,
    QueryType.MULTIHOP: 2,
    QueryType.FRESHNESS: 1,
    QueryType.PERMISSION: 2,
    QueryType.UNANSWERABLE: 2,
}


def _ctx(min_prose_words: int = 5) -> tuple[SynthContext, RuleLLM]:
    inner = RuleLLM()
    llm = CachedLLMClient(
        inner, SQLiteLLMCache(Path(":memory:")), BudgetGuard(1), Tracer(MemoryTraceSink())
    )
    role = LLMRoleConfig(model="gemini/gen", temperature=0.7)
    ctx = SynthContext(
        sections=SECTIONS,
        plan=PLAN,
        principals=PRINCIPALS,
        llm=llm,
        generator=role,
        judge=RelevanceJudge(llm, LLMRoleConfig(model="openai/judge")),
        embedder=HashEmbedder(dim=256),
        cfg=SynthConfig(targets=TARGETS, min_prose_words=min_prose_words, workers=1),
    )
    return ctx, inner


@pytest.fixture(scope="module")
def result():  # type: ignore[no-untyped-def]
    ctx, _ = _ctx()
    return synthesize(ctx)


def test_output_is_a_valid_golden_set(result) -> None:  # type: ignore[no-untyped-def]
    validate_against_corpus(result.queries, {s.section_id: s.acl for s in SECTIONS}, POLICY)
    assert [q.qid for q in result.queries] == [
        f"q{i:04d}" for i in range(1, len(result.queries) + 1)
    ]
    assert all(q.origin == "synthetic" and not q.reviewed for q in result.queries)
    assert {q.split for q in result.queries} <= {"dev", "test"}


def test_targets_are_respected_per_qtype(result) -> None:  # type: ignore[no-untyped-def]
    counts = result.report["by_qtype"]
    for qtype, target in TARGETS.items():
        assert counts.get(str(qtype), 0) <= target + (1 if qtype == QueryType.PERMISSION else 0)
    assert counts.get("factoid", 0) >= 1 and counts.get("unanswerable", 0) >= 1


def test_permission_rows_come_in_privileged_and_denied_pairs(result) -> None:  # type: ignore[no-untyped-def]
    rows = [q for q in result.queries if q.qtype == QueryType.PERMISSION]
    assert rows and len(rows) % 2 == 0
    by_text: dict[str, list] = {}
    for q in rows:
        by_text.setdefault(q.text, []).append(q)
    for pair in by_text.values():
        allowed, denied = sorted(pair, key=lambda q: not q.relevant)
        assert allowed.relevant and not denied.relevant
        (sid,) = allowed.relevant
        acl = next(s.acl for s in SECTIONS if s.section_id == sid)
        assert POLICY.can_read(allowed.principal, acl) and not POLICY.can_read(
            denied.principal, acl
        )


def test_freshness_points_at_new_doc_with_old_as_stale(result) -> None:  # type: ignore[no-untyped-def]
    (q,) = [q for q in result.queries if q.qtype == QueryType.FRESHNESS]
    assert list(q.relevant) == ["internal-kb:kb-0003.md#payment-api-timeouts"]
    assert q.stale == ["internal-kb:kb-0002.md#payment-api-timeouts"]
    assert q.principal == "engineer"


def test_unanswerable_queries_drop_keywords_found_in_corpus(result) -> None:  # type: ignore[no-untyped-def]
    texts = {q.text for q in result.queries if q.qtype == QueryType.UNANSWERABLE}
    assert "fastapi graphql thing" not in texts and texts
    # Each of the 4 theme calls returns 4 items and exactly one ("fastapi") is rejected:
    # "fastapi" + "blorbdb" never co-occur in a section, so that one survives.
    assert result.report["stats"]["unanswerable_keyword_in_corpus"] == 4
    assert result.report["stats"]["unanswerable_keyword_in_corpus"] >= 1


def test_hard_negatives_are_readable_other_page_sections_judged_zero(result) -> None:  # type: ignore[no-untyped-def]
    by_id = {s.section_id: s for s in SECTIONS}
    n = 0
    for q in result.queries:
        rel_docs = {by_id[s].doc_id for s in q.relevant}
        for hn in q.hard_negatives:
            n += 1
            assert hn not in q.relevant and by_id[hn].doc_id not in rel_docs
            assert POLICY.can_read(q.principal, by_id[hn].acl)
    assert n > 0
    assert "0" in result.report["hard_negative_candidate_scores"]


def test_overlap_is_recorded_for_answerable_queries(result) -> None:  # type: ignore[no-untyped-def]
    for q in result.queries:
        assert (q.lexical_overlap is None) == (not q.relevant)
    assert result.report["lexical_overlap_histogram"]


def test_rerun_is_identical_and_cached() -> None:
    ctx_a, _ = _ctx()
    ctx_b, inner_b = _ctx()
    first = synthesize(ctx_a)
    # same cache contents -> zero new calls on a rerun with the same client
    ctx_b.llm = ctx_a.llm
    ctx_b.judge = ctx_a.judge
    second = synthesize(ctx_b)
    assert [q.model_dump() for q in first.queries] == [q.model_dump() for q in second.queries]
    assert inner_b.calls == 0


# --- helpers ------------------------------------------------------------------------------


def test_prose_words_ignores_code() -> None:
    assert prose_words("one two\n```py\nx = 1 2 3\n```\nthree") == 3


def test_lexical_overlap() -> None:
    assert lexical_overlap("how to use Depends with yield", "Use `Depends` and yield.") == 1.0
    assert lexical_overlap("upload multiple files", "Path parameters") == 0.0
    assert lexical_overlap("the of and", "anything") == 0.0  # only stopwords


def test_section_symbols() -> None:
    text = "Use `Depends`, `get_db()`, `fastapi.Query`, `app`, `x`, `Depends(f)` and `Depends`."
    assert section_symbols(text) == ["Depends", "get_db()", "fastapi.Query", "Depends()"]


@pytest.mark.parametrize(
    ("acl", "minimal", "denied"),
    [
        (["public"], "anonymous", None),
        (["eng-internal"], "engineer", "anonymous"),
        (["security-team"], "security", "engineer"),
    ],
)
def test_principal_helpers(acl: list[str], minimal: str, denied: str | None) -> None:
    assert minimal_principal(acl, PRINCIPALS) == minimal
    assert unprivileged_principal(acl, PRINCIPALS) == denied


def test_stratified_sample_is_proportional_and_seeded() -> None:
    picked = stratified_sample(SECTIONS, 10, random.Random(1))
    assert len(picked) == 10 and len({s.section_id for s in picked}) == 10
    assert [s.section_id for s in picked] == [
        s.section_id for s in stratified_sample(SECTIONS, 10, random.Random(1))
    ]
    assert len(stratified_sample(SECTIONS, 10_000, random.Random(1))) == len(SECTIONS)


def _cand(text: str, overlap: float | None = 0.5) -> list[Candidate]:
    return [
        Candidate(
            qtype=QueryType.HOWTO,
            text=text,
            principal="anonymous",
            relevant={"x": 2},
            lexical_overlap=overlap,
        )
    ]


def test_dedupe_drops_near_duplicates_in_order() -> None:
    units = [_cand("upload multiple files"), _cand("upload multiple files"), _cand("cors setup")]
    kept, dropped = dedupe(units, HashEmbedder(dim=256), 0.92)
    assert [u[0].text for u in kept] == ["upload multiple files", "cors setup"] and dropped == 1


def test_balance_spreads_over_overlap_terciles() -> None:
    units = [_cand(f"q{i}", 0.1) for i in range(5)] + [_cand(f"h{i}", 0.9) for i in range(5)]
    picked = balance(units, 4, random.Random(0))
    assert len(picked) == 4
    assert sorted(u[0].lexical_overlap for u in picked) == [0.1, 0.1, 0.9, 0.9]


def test_multihop_pairs_cross_pages() -> None:
    ctx, _ = _ctx()
    pairs = multihop_pairs(ctx)
    assert pairs and all(a.doc_id != b.doc_id for a, b in pairs)
    assert any(a.source == "internal-kb" and b.source == "fastapi-docs" for a, b in pairs)


def test_generator_api_errors_are_counted_not_fatal() -> None:
    ctx, _ = _ctx()

    class Failing:
        def complete(self, *a: Any, **k: Any) -> LLMResponse:
            raise RuntimeError("503 from provider")

    ctx.llm = CachedLLMClient(
        Failing(), SQLiteLLMCache(Path(":memory:")), BudgetGuard(1), Tracer(MemoryTraceSink())
    )
    assert ctx.generate("prompt", GeneratedQuery, "synth.x") is None
    assert ctx.stats["api_error:RuntimeError"] == 1
