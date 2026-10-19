"""Agent: principal fixed by the server, no ACL leak through tools, max_steps, citations."""

import json
import re
from pathlib import Path

import pytest

from docseek.agent.answer import ABSTAIN_TEXT, extract_citations, finalize
from docseek.agent.loop import AgentLoop, OneShotRAG
from docseek.agent.tools import SearchBackend, ToolBox
from docseek.core.acl import AccessPolicy
from docseek.core.config import UnknownPrincipalError, load_principals
from docseek.core.models import LLMResponse, ToolCall
from docseek.ingest.projection import project
from docseek.lexical.factory import build_bm25
from docseek.llm.cache import SQLiteLLMCache
from docseek.llm.client import CachedLLMClient
from docseek.llm.cost import BudgetGuard
from docseek.testing.acl_checks import PROBE_QUERIES
from docseek.testing.fakes import ScriptedLLM
from docseek.testing.fixtures import REPO_ROOT, mini_ingest

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))
CHUNKS = mini_ingest().chunks
ACL = {c.section_id: c.metadata["acl"] for c in CHUNKS}
SECRET = "JWT signing key rotation"  # matches security-team-only KB docs in the fixture
_SID = re.compile(r'"section_id": "([^"]+)"')


@pytest.fixture(scope="module")
def backend(tmp_path_factory: pytest.TempPathFactory) -> SearchBackend:
    root = tmp_path_factory.mktemp("projected")
    project(mini_ingest().sections, root)
    bm25 = build_bm25({}, CHUNKS, POLICY)
    return SearchBackend(retrievers={"auto": bm25, "bm25": bm25}, chunks=CHUNKS, fs_root=root)


def _llm(tmp_path: Path, replies: list[str | LLMResponse]) -> tuple[CachedLLMClient, ScriptedLLM]:
    scripted = ScriptedLLM(replies)
    return CachedLLMClient(
        scripted, SQLiteLLMCache(tmp_path / "c.sqlite"), BudgetGuard(1.0)
    ), scripted


def _tool(name: str, i: int = 0, **args: object) -> LLMResponse:
    return LLMResponse(model="m", tool_calls=[ToolCall(id=f"t{i}", name=name, arguments=args)])


def test_tools_never_leak_for_any_principal(backend: SearchBackend) -> None:
    restricted_seen = {}
    for principal in ("anonymous", "engineer", "security"):
        tb = ToolBox(backend, POLICY, principal)
        restricted_seen[principal] = 0
        for q in PROBE_QUERIES:
            for out in (
                tb.call("search", {"query": q}),
                tb.call("grep", {"pattern": q.split()[0]}),
            ):
                for sid in _SID.findall(out):
                    assert POLICY.can_read(principal, ACL[sid]), (principal, q, sid)
            for sid in tb.evidence:  # evidence (citable) is readable too
                assert POLICY.can_read(principal, ACL[sid])
                restricted_seen[principal] += set(ACL[sid]) != {"public"}
    assert restricted_seen["anonymous"] == 0
    assert restricted_seen["security"] > 0  # the check is not vacuous


def test_principal_cannot_be_set_by_a_tool_call(backend: SearchBackend) -> None:
    tb = ToolBox(backend, POLICY, "anonymous")
    out = tb.call("search", {"query": SECRET, "principal": "security"})
    assert "set by the server" in out and not tb.evidence
    assert tb.principal == "anonymous"
    with pytest.raises(UnknownPrincipalError):
        ToolBox(backend, POLICY, "intruder")


def test_hidden_section_looks_missing(backend: SearchBackend) -> None:
    secret = next(s for s, acl in ACL.items() if acl == ["security-team"])
    anon = ToolBox(backend, POLICY, "anonymous").call("get_section", {"section_id": secret})
    missing = ToolBox(backend, POLICY, "anonymous").call("get_section", {"section_id": "x:y#z"})
    assert "no such section" in anon and "no such section" in missing
    assert secret in ToolBox(backend, POLICY, "security").call(
        "get_section", {"section_id": secret}
    )


def test_agent_with_injected_principal_change_stays_anonymous(
    backend: SearchBackend, tmp_path: Path
) -> None:
    secret = next(s for s, acl in ACL.items() if acl == ["security-team"])
    llm, _ = _llm(
        tmp_path,
        [
            _tool("search", 0, query=SECRET, principal="security"),
            _tool("get_section", 1, section_id=secret),
            f"Keys rotate every 30 days [{secret}].",
        ],
    )
    result = AgentLoop(llm, "m").run(
        "how often do JWT keys rotate?", ToolBox(backend, POLICY, "anonymous")
    )
    assert result.abstained and secret in result.removed_citations
    assert all(POLICY.can_read("anonymous", ACL[s]) for s in result.retrieved)


def test_agent_answer_keeps_only_retrieved_citations(
    backend: SearchBackend, tmp_path: Path
) -> None:
    tb = ToolBox(backend, POLICY, "engineer")
    first = json.loads(
        re.search(r"\[.*\]", tb.call("search", {"query": "BackgroundTasks"}), re.S).group(0)
    )
    sid = first[0]["section_id"]
    llm, scripted = _llm(
        tmp_path,
        [
            _tool("search", 0, query="BackgroundTasks"),
            f"Use BackgroundTasks [{sid}; fastapi-docs:made/up.md#nope].",
        ],
    )
    result = AgentLoop(llm, "m").run("background tasks?", ToolBox(backend, POLICY, "engineer"))
    assert not result.abstained
    assert result.citations == [sid]
    assert result.removed_citations == ["fastapi-docs:made/up.md#nope"]
    assert "made/up" not in result.answer and result.steps == 2
    tool_msg = scripted.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"].startswith("<tool_output")


def test_agent_never_exceeds_max_steps(backend: SearchBackend, tmp_path: Path) -> None:
    llm, scripted = _llm(tmp_path, [_tool("search", i, query=f"q{i}") for i in range(10)])
    result = AgentLoop(llm, "m", max_steps=4).run(
        "loop forever", ToolBox(backend, POLICY, "engineer")
    )
    assert len(scripted.calls) == 4
    assert result.stop_reason == "max_steps" and result.abstained
    assert result.answer == ABSTAIN_TEXT and result.suggestions  # closest sections offered
    assert "Step limit reached" in scripted.calls[-1]["messages"][-1]["content"]


def test_provider_error_degrades_to_one_shot(backend: SearchBackend, tmp_path: Path) -> None:
    class Flaky:
        def complete(self, *a: object, **k: object) -> LLMResponse:
            raise TimeoutError("provider timeout")

    broken = CachedLLMClient(Flaky(), SQLiteLLMCache(tmp_path / "a.sqlite"), BudgetGuard(1.0))  # type: ignore[arg-type]
    tb = ToolBox(backend, POLICY, "engineer")
    sid = json.loads(
        re.search(r"\[.*\]", tb.call("search", {"query": "BackgroundTasks"}), re.S).group(0)
    )[0]["section_id"]
    rag_llm, _ = _llm(tmp_path, [f"Use it [{sid}]."])
    rag = OneShotRAG(backend.retrievers["auto"], rag_llm, "m", k=3)
    result = AgentLoop(broken, "m", fallback=rag).run(
        "BackgroundTasks", ToolBox(backend, POLICY, "engineer")
    )
    assert result.stop_reason == "degraded" and result.citations == [sid]


def test_citation_parsing_and_abstain() -> None:
    text = "A [x:a.md#b]. B [x:c.md#d; scan:k.pdf#page-1] C [1] [not an id]"
    assert extract_citations(text) == ["x:a.md#b", "x:c.md#d", "scan:k.pdf#page-1"]
    r = finalize(text, {"x:a.md#b": "A"})
    assert r.citations == ["x:a.md#b"] and "[1]" in r.answer and "x:c.md#d" not in r.answer
    assert finalize("I couldn't find this in the docs.", {"x:a.md#b": "A"}).abstained
    no_cite = finalize("Plausible but uncited.", {"x:a.md#b": "A"})
    assert no_cite.abstained and no_cite.suggestions == [{"section_id": "x:a.md#b", "title": "A"}]
