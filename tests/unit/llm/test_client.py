"""LLM clients without network: LiteLLM's ``mock_response`` and scripted inner clients."""

import functools
import os
from pathlib import Path
from typing import Any

import litellm
import pytest
from pydantic import BaseModel, SecretStr

from docseek.core.config import Settings
from docseek.core.models import LLMResponse
from docseek.core.tracing import MemoryTraceSink, Tracer, trace_scope
from docseek.llm.cache import SQLiteLLMCache
from docseek.llm.client import (
    CachedLLMClient,
    LiteLLMClient,
    StructuredOutputError,
    build_llm_client,
    parse_structured,
)
from docseek.llm.cost import BudgetExceededError, BudgetGuard
from docseek.testing.fakes import ScriptedLLM

MSG = [{"role": "user", "content": "Rate this."}]
GEMINI = "gemini/gemini-3.8-flash"
OPENAI = "openai/gpt-5.4-mini"


class Verdict(BaseModel):
    score: int
    reason: str


def _mock(content: str) -> Any:
    return functools.partial(litellm.completion, mock_response=content)


def test_price_map_is_local() -> None:
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"


def test_litellm_client_maps_response_usage_and_cost() -> None:
    client = LiteLLMClient({"gemini": "k"}, completion=_mock("hello"))
    resp = client.complete(MSG, model=GEMINI, temperature=0.0)
    assert resp.content == "hello"
    assert resp.model == GEMINI
    assert resp.tokens_in > 0 and resp.tokens_out > 0
    assert resp.cost_usd > 0  # priced from LiteLLM's bundled map
    assert resp.cached is False and resp.parsed is None


def test_litellm_client_parses_structured_output() -> None:
    client = LiteLLMClient(completion=_mock('{"score": 2, "reason": "direct"}'))
    resp = client.complete(MSG, model=OPENAI, response_format=Verdict)
    assert resp.parsed == {"score": 2, "reason": "direct"}


def test_litellm_client_keeps_invalid_structured_reply_unparsed() -> None:
    client = LiteLLMClient(completion=_mock("not json"))
    resp = client.complete(MSG, model=OPENAI, response_format=Verdict)
    assert resp.parsed is None and resp.content == "not json"


def test_litellm_client_passes_provider_key_retries_and_timeout() -> None:
    seen: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return litellm.completion(**kwargs, mock_response="ok")

    client = LiteLLMClient({"gemini": "g-key"}, num_retries=5, timeout_s=9, completion=capture)
    client.complete(MSG, model=GEMINI, temperature=0.7)
    client.complete(MSG, model=OPENAI)
    assert seen[0]["api_key"] == "g-key" and "api_key" not in seen[1]
    assert (seen[0]["num_retries"], seen[0]["timeout"], seen[0]["temperature"]) == (5, 9, 0.7)
    assert "tools" not in seen[0] and "response_format" not in seen[0]


def test_litellm_client_parses_tool_calls() -> None:
    def fake(**kwargs: Any) -> Any:
        return litellm.ModelResponse(
            model="gpt-5.4-mini",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"query": "a"}'},
                            }
                        ],
                    },
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        )

    resp = LiteLLMClient(completion=fake).complete(MSG, model=OPENAI, tools=[{"type": "x"}])
    assert [(t.id, t.name, t.arguments) for t in resp.tool_calls] == [
        ("call_1", "search", {"query": "a"})
    ]
    assert (resp.tokens_in, resp.tokens_out) == (100, 10)


def test_litellm_client_requires_model() -> None:
    with pytest.raises(ValueError, match="explicit model"):
        LiteLLMClient(completion=_mock("x")).complete(MSG)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"score": 1, "reason": "r"}', {"score": 1, "reason": "r"}),
        ('```json\n{"score": 0, "reason": "r"}\n```', {"score": 0, "reason": "r"}),
    ],
)
def test_parse_structured(content: str, expected: dict[str, Any]) -> None:
    assert parse_structured(content, Verdict) == expected


@pytest.mark.parametrize("content", [None, "nope", '{"score": "high"}'])
def test_parse_structured_rejects_bad_replies(content: str | None) -> None:
    with pytest.raises(StructuredOutputError):
        parse_structured(content, Verdict)


def _cached(
    replies: list[str | LLMResponse], limit: float = 1.0
) -> tuple[CachedLLMClient, ScriptedLLM, MemoryTraceSink]:
    inner = ScriptedLLM(replies)
    sink = MemoryTraceSink()
    client = CachedLLMClient(
        inner, SQLiteLLMCache(Path(":memory:")), BudgetGuard(limit), Tracer(sink)
    )
    return client, inner, sink


def test_cached_client_hits_cache_on_identical_call() -> None:
    reply = LLMResponse(content="a", model=OPENAI, tokens_in=7, tokens_out=3, cost_usd=0.02)
    client, inner, sink = _cached([reply])
    with trace_scope("t1"):
        first = client.complete(MSG, model=OPENAI, name="synth.query")
        second = client.complete(MSG, model=OPENAI, name="synth.query")
    assert len(inner.calls) == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.content == "a"
    assert client.budget.spent_usd == pytest.approx(0.02)
    assert [(e.trace_id, e.step, e.name, e.cost_usd) for e in sink.events] == [
        ("t1", 0, "synth.query", 0.02),
        ("t1", 1, "synth.query", 0.0),
    ]
    assert sink.events[1].output == {"cached": True, "original_cost_usd": 0.02, "n_tool_calls": 0}
    assert sink.events[0].tokens_in == 7


def test_cached_client_separates_temperatures_and_models() -> None:
    client, inner, _ = _cached(["a", "b", "c"])
    client.complete(MSG, model=OPENAI, temperature=0.0)
    client.complete(MSG, model=OPENAI, temperature=0.7)
    client.complete(MSG, model=GEMINI, temperature=0.0)
    assert len(inner.calls) == 3


def test_budget_blocks_new_calls_but_not_cache_hits() -> None:
    client, inner, _ = _cached([LLMResponse(content="a", model=OPENAI, cost_usd=1.5)], limit=1.0)
    client.complete(MSG, model=OPENAI)
    assert client.complete(MSG, model=OPENAI).cached  # still free
    with pytest.raises(BudgetExceededError):
        client.complete([{"role": "user", "content": "new"}], model=OPENAI)
    assert len(inner.calls) == 1


def test_invalid_structured_reply_is_paid_traced_but_not_cached() -> None:
    bad = LLMResponse(content="nope", model=OPENAI, cost_usd=0.01)
    good = LLMResponse(content='{"score": 2, "reason": "ok"}', model=OPENAI, cost_usd=0.01)

    class Inner:
        def __init__(self) -> None:
            self.replies = [bad, good]

        def complete(self, messages: Any, **kw: Any) -> LLMResponse:
            r = self.replies.pop(0)
            if r.content and r.content.startswith("{"):
                return r.model_copy(update={"parsed": parse_structured(r.content, Verdict)})
            return r

    sink = MemoryTraceSink()
    client = CachedLLMClient(
        Inner(), SQLiteLLMCache(Path(":memory:")), BudgetGuard(1), Tracer(sink)
    )
    with pytest.raises(StructuredOutputError):
        client.complete(MSG, model=OPENAI, response_format=Verdict)
    assert client.budget.spent_usd == pytest.approx(0.01)
    assert len(client.cache) == 0
    resp = client.complete(MSG, model=OPENAI, response_format=Verdict)
    assert resp.parsed == {"score": 2, "reason": "ok"}
    assert client.complete(MSG, model=OPENAI, response_format=Verdict).cached
    assert [e.cost_usd for e in sink.events] == [0.01, 0.01, 0.0]


def test_cached_client_requires_model() -> None:
    client, _, _ = _cached(["a"])
    with pytest.raises(ValueError, match="model is required"):
        client.complete(MSG)


def test_build_llm_client_wires_settings(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        gemini_api_key=SecretStr("g"),
        openai_api_key=SecretStr(" "),
        DOCSEEK_LLM_CACHE=tmp_path / "c.sqlite",
        DOCSEEK_BUDGET_USD=2.5,
    )
    seen: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return litellm.completion(**kwargs, mock_response="ok")

    client = build_llm_client(settings, completion=capture)
    assert client.budget.limit_usd == 2.5
    client.complete(MSG, model=GEMINI)
    client.complete(MSG, model=OPENAI)
    assert seen[0]["api_key"] == "g" and "api_key" not in seen[1]  # blank key is "not set"
    assert (tmp_path / "c.sqlite").is_file()


def test_litellm_retries_work_offline() -> None:
    """Regression: LiteLLM's retry path imports tenacity; a 503 must surface as the
    provider error after retrying, not as an ImportError (seen in the first synth run)."""
    error = litellm.InternalServerError(message="503", llm_provider="gemini", model="m")
    client = LiteLLMClient(
        num_retries=1, completion=functools.partial(litellm.completion, mock_response=error)
    )
    with pytest.raises(litellm.InternalServerError):
        client.complete(MSG, model=GEMINI)
