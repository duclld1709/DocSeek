"""Project spend ledger, per-model parameters, rpm limiter and their wiring (offline)."""

import threading
from pathlib import Path
from typing import Any

import litellm
import pytest
from pydantic import SecretStr

from docseek.core.config import LLMConfig, LLMRoleConfig, ModelChoice, ModelsConfig, Settings
from docseek.core.models import LLMResponse
from docseek.core.tracing import MemoryTraceSink, Tracer
from docseek.llm.cache import SQLiteLLMCache, cache_key
from docseek.llm.client import CachedLLMClient, LiteLLMClient, RateLimiter, build_llm_client
from docseek.llm.cost import BudgetExceededError, BudgetGuard, SpendLedger
from docseek.testing.fakes import ScriptedLLM

MSG = [{"role": "user", "content": "hi"}]
OPENAI, GEMINI = "openai/nano", "gemini/lite"


def test_ledger_records_totals_and_summary(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "c.sqlite")
    ledger.record(model=OPENAI, name="judge", tokens_in=10, tokens_out=2, cost_usd=0.25)
    ledger.record(model=OPENAI, name="judge", tokens_in=5, tokens_out=1, cost_usd=0.5)
    ledger.record(model=GEMINI, name="synth", tokens_in=7, tokens_out=3, cost_usd=0.1)
    ledger.close()
    reopened = SpendLedger(tmp_path / "c.sqlite")  # persists across runs
    assert reopened.total("openai") == pytest.approx(0.75)
    assert reopened.total() == pytest.approx(0.85)
    assert reopened.summary()["openai"] == {
        "calls": 2,
        "tokens_in": 15,
        "tokens_out": 3,
        "cost_usd": pytest.approx(0.75),
    }
    reopened.check(GEMINI, {"openai": 0.75})  # other providers unaffected
    with pytest.raises(BudgetExceededError, match="project spend limit for openai"):
        reopened.check(OPENAI, {"openai": 0.75})


def _cached(replies: list[LLMResponse], limits: dict[str, float]) -> CachedLLMClient:
    return CachedLLMClient(
        ScriptedLLM(list(replies)),
        SQLiteLLMCache(Path(":memory:")),
        BudgetGuard(100),
        Tracer(MemoryTraceSink()),
        ledger=SpendLedger(Path(":memory:")),
        spend_limits=limits,
    )


def test_cached_client_stops_at_the_project_limit_and_ignores_cache_hits() -> None:
    paid = [LLMResponse(content=str(i), model=OPENAI, cost_usd=0.4) for i in range(5)]
    client = _cached(paid, {"openai": 1.0})
    for i in range(3):
        client.complete([{"role": "user", "content": str(i)}], model=OPENAI)
    assert client.ledger is not None and client.ledger.total("openai") == pytest.approx(1.2)
    assert client.complete([{"role": "user", "content": "0"}], model=OPENAI).cached
    with pytest.raises(BudgetExceededError, match="project spend limit"):
        client.complete([{"role": "user", "content": "new"}], model=OPENAI)
    assert client.ledger.summary()["openai"]["calls"] == 3


def test_model_params_are_sent_and_key_the_cache() -> None:
    seen: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return litellm.completion(**kwargs, mock_response="ok")

    params = {"openai/gpt-5.4-nano": {"reasoning_effort": "none"}}
    LiteLLMClient(model_params=params, completion=capture).complete(
        MSG, model="openai/gpt-5.4-nano"
    )
    assert seen[0]["reasoning_effort"] == "none"
    base = {
        "model": OPENAI,
        "messages": MSG,
        "tools": None,
        "temperature": 0.0,
        "response_format": None,
    }
    assert cache_key(**base) == cache_key(**base, extra={})
    assert cache_key(**base, extra={"reasoning_effort": "none"}) != cache_key(**base)
    assert cache_key(**base, extra={"reasoning_effort": "low"}) != cache_key(
        **base, extra={"reasoning_effort": "none"}
    )


def test_rate_limiter_spaces_calls() -> None:
    now = [100.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    limiter = RateLimiter(30, clock=lambda: now[0], sleep=sleep)  # one call every 2 s
    for _ in range(3):
        limiter.wait()
    assert slept == [pytest.approx(2.0), pytest.approx(2.0)]
    now[0] += 10
    limiter.wait()
    assert len(slept) == 2  # idle long enough: no wait
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_rate_limiter_is_thread_safe() -> None:
    now = [0.0]
    lock = threading.Lock()
    total: list[float] = []

    def sleep(s: float) -> None:
        with lock:
            total.append(s)

    limiter = RateLimiter(60, clock=lambda: now[0], sleep=sleep)
    threads = [threading.Thread(target=limiter.wait) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(total) == [pytest.approx(x) for x in (1.0, 2.0, 3.0, 4.0)]


def _models(**judge: Any) -> ModelsConfig:
    gen = LLMRoleConfig(model=GEMINI, temperature=0.7, reasoning_effort="none", rpm=10)
    return ModelsConfig(
        llm=LLMConfig(
            generator=gen,
            judge=LLMRoleConfig(model=OPENAI, **judge),
            agent=gen.model_copy(update={"rpm": 5}),
            vision=gen,
        ),
        embedding=ModelChoice(default="e"),
        reranker=ModelChoice(default="r"),
        spend_limits_usd={"openai": 2.0},
    )


def test_models_config_params_and_rpm() -> None:
    cfg = _models(reasoning_effort="none")
    assert cfg.model_params() == {
        GEMINI: {"reasoning_effort": "none"},
        OPENAI: {"reasoning_effort": "none"},
    }
    assert cfg.rpm_limits() == {GEMINI: 5}  # strictest role wins
    gen = cfg.llm.generator
    clash = cfg.model_copy(
        update={
            "llm": cfg.llm.model_copy(
                update={"agent": gen.model_copy(update={"reasoning_effort": "low"})}
            )
        }
    )
    with pytest.raises(ValueError, match="disagree"):
        clash.model_params()


def test_committed_models_config_disables_thinking_and_caps_openai(configs_dir: Path) -> None:
    from docseek.core.config import load_models_config

    cfg = load_models_config(configs_dir)
    assert cfg.llm.generator.provider != cfg.llm.judge.provider
    assert cfg.llm.judge.reasoning_effort == "none" and cfg.llm.judge.temperature == 0.0
    assert cfg.spend_limits_usd["openai"] <= 2.0


def test_build_llm_client_wires_ledger_limits_and_params(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key=SecretStr("o"),
        DOCSEEK_LLM_CACHE=tmp_path / "c.sqlite",
    )
    seen: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return litellm.completion(**kwargs, mock_response="ok")

    client = build_llm_client(settings, models=_models(reasoning_effort="none"), completion=capture)
    client.complete(MSG, model=OPENAI)
    assert seen[0]["reasoning_effort"] == "none" and seen[0]["api_key"] == "o"
    assert client.spend_limits == {"openai": 2.0}
    assert client.ledger is not None and client.ledger.summary()["openai"]["calls"] == 1
