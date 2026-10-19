"""API with TestClient + ScriptedLLM on the mini corpus: ACL, principal header, traces."""

from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from docseek.agent.runtime import Runtime, ServiceConfig
from docseek.agent.tools import SearchBackend
from docseek.api.app import create_app
from docseek.core.acl import AccessPolicy
from docseek.core.config import load_principals
from docseek.core.models import LLMResponse, ToolCall
from docseek.core.tracing import Tracer
from docseek.lexical.factory import build_bm25
from docseek.llm.cache import SQLiteLLMCache
from docseek.llm.client import CachedLLMClient
from docseek.llm.cost import BudgetGuard
from docseek.testing.fakes import ScriptedLLM
from docseek.testing.fixtures import REPO_ROOT, mini_ingest

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))
CHUNKS = mini_ingest().chunks
ACL = {c.section_id: c.metadata["acl"] for c in CHUNKS}
SECRET = "JWT signing key rotation"


def _client(tmp_path: Path, replies: list[str | LLMResponse]) -> TestClient:
    bm25 = build_bm25({}, CHUNKS, POLICY)
    backend = SearchBackend(retrievers={"auto": bm25, "bm25": bm25}, chunks=CHUNKS)
    cfg = ServiceConfig(chunks_file=Path("x"), search_modes={}, one_shot={})
    rt = Runtime(cfg, POLICY, backend, bm25)

    def llm(tracer: Tracer) -> CachedLLMClient:
        cache = SQLiteLLMCache(tmp_path / "c.sqlite")
        return CachedLLMClient(ScriptedLLM(replies), cache, BudgetGuard(1.0), tracer=tracer)

    return TestClient(create_app(lambda: rt, llm, "scripted"))


def test_health(tmp_path: Path) -> None:
    r = _client(tmp_path, []).get("/health")
    assert r.status_code == 200 and r.json()["chunks"] == len(CHUNKS)


@pytest.mark.parametrize("principal", ["anonymous", "engineer", "security"])
def test_search_enforces_acl(tmp_path: Path, principal: str) -> None:
    client = _client(tmp_path, [])
    r = client.post("/search", json={"query": SECRET, "k": 20}, headers={"X-Principal": principal})
    assert r.status_code == 200
    sids = [h["section_id"] for h in r.json()["hits"]]
    assert all(POLICY.can_read(principal, ACL[s]) for s in sids)
    restricted = [s for s in sids if "security-team" in ACL[s]]
    assert bool(restricted) == (principal == "security")


def test_principal_comes_from_header(tmp_path: Path) -> None:
    client = _client(tmp_path, [])
    body = {"query": SECRET, "principal": "security"}
    assert client.post("/search", json=body).status_code == 403  # header says anonymous
    assert (
        client.post("/search", json={"query": "x"}, headers={"X-Principal": "root"}).status_code
        == 403
    )
    ok = client.post("/search", json=body, headers={"X-Principal": "security"})
    assert ok.status_code == 200
    assert client.post("/search", json={"query": "x", "mode": "nope"}).status_code == 422


def test_ask_cites_retrieved_and_traces_are_private(tmp_path: Path) -> None:
    probe = _client(tmp_path / "p", []).post("/search", json={"query": "BackgroundTasks"})
    sid = probe.json()["hits"][0]["section_id"]
    replies: list[str | LLMResponse] = [
        LLMResponse(
            model="scripted",
            tool_calls=[ToolCall(id="t0", name="search", arguments={"query": "BackgroundTasks"})],
        ),
        f"Use BackgroundTasks [{sid}] [fastapi-docs:fake.md#x].",
    ]
    client = _client(tmp_path, replies)
    r = client.post("/ask", json={"query": "background tasks?", "debug": True})
    data = r.json()
    assert r.status_code == 200 and data["citations"] == [sid]
    assert data["removed_citations"] == ["fastapi-docs:fake.md#x"]
    assert data["tool_calls"] and data["principal"] == "anonymous"
    trace = client.get(f"/traces/{data['trace_id']}")
    kinds = [e["kind"] for e in trace.json()["events"]]
    assert {"llm_call", "tool_call", "answer"} <= set(kinds)
    other = client.get(f"/traces/{data['trace_id']}", headers={"X-Principal": "security"})
    assert other.status_code == 404  # traces are private to their principal
    assert client.get("/traces/doesnotexist").status_code == 404


def test_section_endpoint_hides_restricted(tmp_path: Path) -> None:
    client = _client(tmp_path, [])
    secret = next(s for s, acl in ACL.items() if acl == ["security-team"])
    url = "/sections/" + quote(secret, safe="")  # '#' in ids must be URL-encoded
    assert client.get(url).status_code == 404
    assert client.get("/sections/" + quote("x:y.md#z", safe="")).status_code == 404
    ok = client.get(url, headers={"X-Principal": "security"})
    assert ok.status_code == 200 and ok.json()["section_id"] == secret
