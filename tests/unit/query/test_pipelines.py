"""M5 wrappers built from configs: Leakage = 0 on every route, including LLM-driven ones."""

from pathlib import Path

import pytest

from docseek.core.acl import AccessPolicy
from docseek.core.config import load_principals
from docseek.core.models import Query
from docseek.eval import pipelines
from docseek.eval.runner import build_child
from docseek.llm.cache import SQLiteLLMCache
from docseek.llm.client import CachedLLMClient
from docseek.llm.cost import BudgetGuard
from docseek.testing.acl_checks import probe_leaks
from docseek.testing.fakes import ScriptedLLM
from docseek.testing.fixtures import REPO_ROOT, mini_ingest

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))
CHUNKS = mini_ingest().chunks
ACL = {c.chunk_id: c.metadata["acl"] for c in CHUNKS}
BM25 = {"type": "bm25", "index_dir": None}
DENSE = {"type": "dense", "embedder": "hash", "location": ":memory:"}


def _build(raw: dict) -> object:
    return build_child(raw, CHUNKS, [], POLICY, 0)


def test_freshness_and_router_paths_never_leak() -> None:
    fresh = _build({"type": "freshness", "inner": BM25})
    assert probe_leaks(fresh, ACL, POLICY).leaks == []
    router = _build(
        {
            "type": "router",
            "use_llm": False,
            "default": {"type": "freshness", "inner": BM25},
            "routes": {"symbol": BM25, "conceptual": DENSE},
        }
    )
    res = probe_leaks(router, ACL, POLICY)
    assert res.leaks == [] and res.restricted_seen["engineer"] > 0
    routes = {d[2] for d in router.decisions.values()}  # type: ignore[attr-defined]
    assert len(routes) >= 2  # probe queries exercise more than one path


def test_filtered_and_expand_never_leak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The LLM tries to steer towards restricted docs; ACL still comes from the principal.
    replies = ['{"doc_type": ["security-policy"]}', '{"queries": ["JWT signing key rotation"]}']
    client = CachedLLMClient(
        ScriptedLLM(replies * 50), SQLiteLLMCache(tmp_path / "c.sqlite"), BudgetGuard(1.0)
    )
    monkeypatch.setattr(pipelines, "llm", lambda: client)
    filtered = _build({"type": "filtered", "inner": BM25})
    expand = _build({"type": "expand", "mode": "rewrite", "n": 1, "inner": BM25})
    for r in (filtered, expand):
        res = probe_leaks(r, ACL, POLICY, queries=["JWT signing key rotation"])
        assert res.leaks == []
    hits = filtered.search(Query(qid="z", text="JWT signing key rotation"), 5)  # type: ignore[attr-defined]
    assert all(POLICY.can_read("anonymous", ACL[h.chunk_id]) for h in hits)
