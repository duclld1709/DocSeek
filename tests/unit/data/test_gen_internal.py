"""gen-internal: plan invariants, body checks, retries and the CLI (scripted LLM only)."""

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docseek.cli import app
from docseek.core.config import LLMRoleConfig, load_corpus_config, load_principals
from docseek.core.tracing import MemoryTraceSink, Tracer
from docseek.data.gen_internal import (
    KBPlan,
    KBPlanEntry,
    build_messages,
    check_body,
    generate_internal_kb,
    key_terms,
    load_plan,
    render_document,
)
from docseek.ingest.internal_kb import load_kb
from docseek.llm.cache import SQLiteLLMCache
from docseek.llm.client import CachedLLMClient
from docseek.llm.cost import BudgetGuard
from docseek.testing.fakes import ScriptedLLM

REPO = Path(__file__).resolve().parents[3]
PLAN = load_plan(REPO / "configs" / "internal_kb_plan.yaml")
ROLE = LLMRoleConfig(model="gemini/test-model", temperature=0.7)
ENTRY = KBPlanEntry(
    id="kb-0001",
    title="Card processor timeouts",
    doc_type="runbook",
    acl=["eng-internal"],
    owner="payments-team",
    version=2,
    updated_at=date(2026, 8, 2),
    facts=["The timeout is 10 seconds.", "Alerts go to `sec-alerts-critical`."],
    fastapi_refs=["Depends"],
)
FILLER = " ".join(["Operators should follow the steps below carefully."] * 20)


def _body(entry: KBPlanEntry = ENTRY, *, extra: str = "") -> str:
    facts = " ".join(entry.facts)
    refs = " ".join(f"`{r}`" for r in entry.fastapi_refs)
    return (
        f"# {entry.title} {{ #title }}\n\n{facts} {refs}\n\n"
        f"## Background {{ #background }}\n\n{FILLER}\n\n"
        f"## Steps {{ #steps }}\n\n{FILLER}\n{extra}"
    )


def _reply(markdown: str) -> str:
    return json.dumps({"markdown": markdown})


def _llm(replies: list[str]) -> tuple[CachedLLMClient, ScriptedLLM]:
    inner = ScriptedLLM(replies)
    client = CachedLLMClient(
        inner, SQLiteLLMCache(Path(":memory:")), BudgetGuard(1.0), Tracer(MemoryTraceSink())
    )
    return client, inner


# --- the committed plan -------------------------------------------------------------------


def test_plan_has_50_docs_with_sequential_unique_ids() -> None:
    ids = [d.id for d in PLAN.docs]
    assert ids == [f"kb-{i:04d}" for i in range(1, 51)]


def test_plan_has_eight_conflicting_supersedes_pairs() -> None:
    by_id = {d.id: d for d in PLAN.docs}
    pairs = [(by_id[d.supersedes], d) for d in PLAN.docs if d.supersedes]
    assert len(pairs) == 8
    for old, new in pairs:
        assert (old.status, new.status) == ("deprecated", "active")
        assert old.title == new.title and old.acl == new.acl
        assert old.updated_at < new.updated_at and old.version < new.version
        assert set(old.facts) != set(new.facts)  # the pair contradicts on at least one fact
    deprecated = {d.id for d in PLAN.docs if d.status == "deprecated"}
    assert deprecated == {old.id for old, _ in pairs}


def test_plan_acl_mix_and_fastapi_references() -> None:
    acl = Counter(g for d in PLAN.docs for g in d.acl)
    assert acl["security-team"] == 6
    assert acl["public"] >= 8 and acl["eng-internal"] >= 20
    assert all(len(d.acl) == 1 for d in PLAN.docs)
    assert sum(bool(d.fastapi_refs) for d in PLAN.docs) >= 10
    assert max(d.updated_at for d in PLAN.docs) <= date(2026, 10, 17)


def test_every_plan_fact_has_something_checkable_or_is_prose() -> None:
    n_checked = sum(bool(key_terms(f)) for d in PLAN.docs for f in d.facts)
    assert n_checked >= 75


def test_plan_renders_into_a_valid_kb(tmp_path: Path) -> None:
    for entry in PLAN.docs:
        doc = render_document(entry, _body(entry))
        (tmp_path / f"{entry.id}.md").write_text(doc, encoding="utf-8")
    cfg = load_corpus_config(REPO / "configs").internal_kb
    docs = load_kb(tmp_path, cfg, load_principals(REPO / "configs"))
    assert len(docs) == 50


# --- key terms and body checks ------------------------------------------------------------


@pytest.mark.parametrize(
    ("fact", "terms"),
    [
        ("The timeout is 10 seconds.", ["10"]),
        ("Escalate to on-call after 15 minutes.", ["15"]),
        ("Use HMAC-SHA256 in Acme-Signature.", ["HMAC-SHA256", "Acme-Signature"]),
        ("Restart `ledger-service` between 23:00-01:00 UTC.", ["ledger-service", "23:00", "01:00"]),
        ("ACME-4001 maps to HTTP 402.", ["ACME-4001", "402"]),
        ("Uptime is 99.95%.", ["99.95%"]),
        ("Incident on 2026-01-14 lasted 6 hours.", ["2026-01-14", "6"]),
        ("Use the starting_after parameter.", ["starting_after"]),
        ("All new services use FastAPI.", []),
    ],
)
def test_key_terms(fact: str, terms: list[str]) -> None:
    assert key_terms(fact) == terms


def test_good_body_passes() -> None:
    assert check_body(ENTRY, _body()) == []


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        (_body().replace("10 seconds", "ten seconds"), "fact not stated exactly (10 missing)"),
        (_body().replace("`sec-alerts-critical`", "the alerts channel"), "sec-alerts-critical"),
        (_body().replace("`Depends`", ""), "`Depends`"),
        (_body().replace("## Steps { #steps }", "## Steps"), "heading without"),
        (_body().replace("## Steps { #steps }", "Steps"), "at least two"),
        ("Intro\n" + _body(), "first line must be the H1"),
        ("# T { #t }\n## A { #a }\n## B { #b }\n10 `sec-alerts-critical` `Depends`", "length"),
    ],
)
def test_bad_bodies_are_flagged(body: str, problem: str) -> None:
    assert any(problem in p for p in check_body(ENTRY, body))


def test_table_is_required_when_planned() -> None:
    entry = ENTRY.model_copy(update={"table": "Include a table."})
    assert any("table" in p for p in check_body(entry, _body()))
    assert check_body(entry, _body(extra="\n| a | b |\n|---|---|\n| 1 | 2 |\n")) == []


def test_hash_lines_in_code_are_not_headings() -> None:
    body = _body(extra="\n```bash\n# not a heading\n```\n")
    assert check_body(ENTRY, body) == []


def test_build_messages_lists_facts_refs_and_feedback() -> None:
    msgs = build_messages(ENTRY, "Acme Payments", ["fact not stated exactly: x"])
    assert "Acme Payments" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert all(f in user for f in ENTRY.facts) and "`Depends`" in user
    assert "fact not stated exactly: x" in msgs[2]["content"]
    assert len(build_messages(ENTRY, "Acme Payments")) == 2


# --- generation loop ----------------------------------------------------------------------


def test_generation_retries_with_feedback_then_writes(tmp_path: Path) -> None:
    llm, inner = _llm([_reply("# Too short { #t }"), _reply(_body())])
    plan = KBPlan(company="Acme Payments", docs=[ENTRY])
    (result,) = generate_internal_kb(plan, llm=llm, role=ROLE, out_dir=tmp_path, workers=1)
    assert (result.status, result.attempts) == ("written", 2)
    assert "previous draft" in inner.calls[1]["messages"][-1]["content"]
    assert inner.calls[0]["model"] == ROLE.model and inner.calls[0]["temperature"] == 0.7
    text = (tmp_path / "kb-0001.md").read_text(encoding="utf-8")
    assert text.startswith("---\nid: kb-0001\n") and "acl: [eng-internal]" in text


def test_generation_skips_existing_and_reports_failures(tmp_path: Path) -> None:
    (tmp_path / "kb-0001.md").write_text("existing", encoding="utf-8")
    second = ENTRY.model_copy(update={"id": "kb-0002"})
    llm, inner = _llm(["not json", _reply("# x { #x }"), _reply("# y { #y }")])
    plan = KBPlan(company="Acme Payments", docs=[ENTRY, second])
    results = generate_internal_kb(plan, llm=llm, role=ROLE, out_dir=tmp_path, workers=1)
    assert [(r.id, r.status) for r in results] == [("kb-0001", "skipped"), ("kb-0002", "failed")]
    assert results[1].attempts == 3 and results[1].problems
    assert (tmp_path / "kb-0001.md").read_text(encoding="utf-8") == "existing"
    assert not (tmp_path / "kb-0002.md").exists()
    assert len(inner.calls) == 3


def test_cli_rejects_unknown_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCSEEK_EXPERIMENTS_DIR", str(tmp_path))
    result = CliRunner().invoke(app, ["data", "gen-internal", "--only", "kb-9999"])
    assert result.exit_code == 1 and "unknown KB ids" in result.output
