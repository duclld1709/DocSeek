"""DeepEval judge adapter goes through CachedLLMClient: schema parsing, cache, budget."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from docseek.core.models import LLMResponse
from docseek.eval.answer_judge import DocSeekJudge
from docseek.llm.cache import SQLiteLLMCache
from docseek.llm.client import CachedLLMClient
from docseek.llm.cost import BudgetExceededError, BudgetGuard
from docseek.testing.fakes import ScriptedLLM


class Verdict(BaseModel):
    verdict: str


def test_judge_parses_schema_and_uses_cache(tmp_path: Path) -> None:
    scripted = ScriptedLLM(['{"verdict": "yes"}', "free text"])
    llm = CachedLLMClient(scripted, SQLiteLLMCache(tmp_path / "c.sqlite"), BudgetGuard(1.0))
    judge = DocSeekJudge(llm, "openai/judge")
    assert judge.generate("claim?", schema=Verdict) == Verdict(verdict="yes")
    assert judge.generate("claim?", schema=Verdict) == Verdict(verdict="yes")  # cache hit
    assert judge.generate("summarise") == "free text"
    assert len(scripted.calls) == 2 and judge.get_model_name() == "openai/judge"


def test_judge_respects_budget(tmp_path: Path) -> None:
    paid = LLMResponse(content='{"verdict": "no"}', model="openai/judge", cost_usd=2.0)
    llm = CachedLLMClient(
        ScriptedLLM([paid, paid]), SQLiteLLMCache(tmp_path / "b.sqlite"), BudgetGuard(1.0)
    )
    judge = DocSeekJudge(llm, "openai/judge")
    judge.generate("first", schema=Verdict)
    with pytest.raises(BudgetExceededError):
        judge.generate("second", schema=Verdict)
