from pathlib import Path

from docseek.core.acl import AccessPolicy
from docseek.core.config import LLMRoleConfig, load_principals
from docseek.lexical import factory
from docseek.llm.cache import SQLiteLLMCache
from docseek.llm.client import CachedLLMClient
from docseek.llm.cost import BudgetGuard
from docseek.query.rewrite import DecomposeRetriever, HyDERetriever, MultiQueryRetriever
from docseek.testing.acl_checks import probe_leaks
from docseek.testing.fakes import ScriptedLLM
from docseek.testing.fixtures import REPO_ROOT, mini_ingest

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))
CHUNKS = mini_ingest().chunks
ACL = {c.chunk_id: c.metadata["acl"] for c in CHUNKS}
ROLE = LLMRoleConfig(model="gemini/x")


def _llm(tmp: Path, replies: list[str]) -> CachedLLMClient:
    return CachedLLMClient(ScriptedLLM(replies), SQLiteLLMCache(tmp / "c.sqlite"), BudgetGuard(1.0))


def test_expansions_never_leak_and_degrade_gracefully(tmp_path: Path) -> None:
    bm25 = factory.build_bm25({"index_dir": None}, CHUNKS, POLICY)
    # Every probe call gets an expansion that targets restricted content.
    evil = '{"queries": ["JWT signing key rotation security-team", "payment runbook"]}'
    n_calls = 3 * 6  # principals x probe queries; identical prompts hit the cache
    mq = MultiQueryRetriever("mq", bm25, _llm(tmp_path, [evil] * n_calls), ROLE, n=2)
    res = probe_leaks(mq, ACL, POLICY)
    assert res.leaks == [] and res.restricted_seen["engineer"] > 0
    hyde = HyDERetriever("hyde", bm25, _llm(tmp_path / "h", ["not json"] * n_calls), ROLE)
    (tmp_path / "h").mkdir(exist_ok=True)
    assert probe_leaks(hyde, ACL, POLICY).leaks == []  # invalid LLM output -> original only
    dec = DecomposeRetriever("dec", bm25, _llm(tmp_path / "d", [evil] * n_calls), ROLE)
    (tmp_path / "d").mkdir(exist_ok=True)
    assert probe_leaks(dec, ACL, POLICY).leaks == []
