"""Cache keys, SQLite persistence and the budget guard."""

import threading
from pathlib import Path

import pytest
from pydantic import BaseModel

from docseek.core.models import LLMResponse
from docseek.llm.cache import SQLiteLLMCache, cache_key
from docseek.llm.cost import BudgetExceededError, BudgetGuard

MSG = [{"role": "user", "content": "hi"}]


class A(BaseModel):
    x: int


class B(BaseModel):
    y: str


def _key(**kw: object) -> str:
    args: dict = {
        "model": "openai/m",
        "messages": MSG,
        "tools": None,
        "temperature": 0.0,
        "response_format": None,
    }
    args.update(kw)
    return cache_key(**args)


def test_cache_key_is_stable_and_sensitive_to_every_input() -> None:
    base = _key()
    assert base == _key() and len(base) == 64
    variants = [
        _key(model="gemini/m"),
        _key(messages=[{"role": "user", "content": "hi!"}]),
        _key(tools=[{"type": "function", "function": {"name": "f"}}]),
        _key(temperature=0.7),
        _key(response_format=A),
        _key(response_format=B),
    ]
    assert len({base, *variants}) == 7


def test_cache_key_ignores_dict_key_order() -> None:
    reordered = [{"content": "hi", "role": "user"}]
    assert _key(messages=reordered) == _key()


def test_sqlite_cache_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "llm.sqlite"
    resp = LLMResponse(content="ok", model="openai/m", tokens_in=3, cost_usd=0.01)
    cache = SQLiteLLMCache(path)
    assert cache.get("k") is None
    cache.put("k", resp)
    cache.close()
    reopened = SQLiteLLMCache(path)
    assert reopened.get("k") == resp
    assert len(reopened) == 1


def test_budget_guard_blocks_once_spent() -> None:
    guard = BudgetGuard(1.0)
    guard.check()
    guard.add(0.6)
    guard.check()
    guard.add(0.6)
    assert guard.spent_usd == pytest.approx(1.2)
    assert guard.remaining_usd == 0.0
    with pytest.raises(BudgetExceededError, match="budget exhausted"):
        guard.check()


def test_budget_guard_is_thread_safe() -> None:
    guard = BudgetGuard(100.0)
    threads = [
        threading.Thread(target=lambda: [guard.add(0.01) for _ in range(1000)]) for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert guard.spent_usd == pytest.approx(80.0)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_budget_guard_rejects_non_positive_limits(bad: float) -> None:
    with pytest.raises(ValueError):
        BudgetGuard(bad)


def test_budget_guard_rejects_negative_costs() -> None:
    with pytest.raises(ValueError):
        BudgetGuard(1.0).add(-0.1)
