from pathlib import Path

import pytest

from docseek.core.config import LLMRoleConfig
from docseek.core.models import QueryType
from docseek.llm.cache import SQLiteLLMCache
from docseek.llm.client import CachedLLMClient
from docseek.llm.cost import BudgetGuard
from docseek.query.classifier import QueryClassifier, rule_type
from docseek.testing.fakes import ScriptedLLM


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("BackgroundTasks", QueryType.SYMBOL),
        ("what does `response_model` do", QueryType.SYMBOL),
        ("@app.get decorator", QueryType.SYMBOL),
        ("OAuth2PasswordBearer tokenUrl", QueryType.SYMBOL),
        ("what is the latest payment timeout policy", QueryType.FRESHNESS),
        ("is the old rotation runbook still valid", QueryType.FRESHNESS),
        ("why use async def", None),
        ("how do I add CORS", None),
    ],
)
def test_rules(text: str, expected: QueryType | None) -> None:
    assert rule_type(text) == expected


def test_llm_path_and_fallbacks(tmp_path: Path) -> None:
    scripted = ScriptedLLM(['{"qtype": "conceptual"}', "not json"])
    llm = CachedLLMClient(scripted, SQLiteLLMCache(tmp_path / "c.sqlite"), BudgetGuard(1.0))
    role = LLMRoleConfig(model="gemini/x", temperature=0.0)
    clf = QueryClassifier(llm, role)
    assert clf.classify("BackgroundTasks").source == "rule"
    out = clf.classify("why use async def")
    assert (out.qtype, out.source) == (QueryType.CONCEPTUAL, "llm")
    assert clf.classify("why use async def").source == "llm"  # cached: no 2nd scripted call
    bad = clf.classify("something else entirely")
    assert (bad.qtype, bad.source) == (QueryType.FACTOID, "fallback")
    offline = QueryClassifier(None, None).classify("how do I add CORS")
    assert (offline.qtype, offline.source) == (QueryType.HOWTO, "fallback")
