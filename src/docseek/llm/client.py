"""LLM clients.

- ``LiteLLMClient`` talks to providers through LiteLLM (retries with backoff, cost from
  LiteLLM's bundled price map).
- ``CachedLLMClient`` wraps any ``LLMClient`` with the SQLite cache, the run budget and
  tracing. **Every LLM call in DocSeek goes through it** (CLAUDE.md); ``build_llm_client``
  is the one place that assembles the stack.

LiteLLM would otherwise download its price map from GitHub at import time, so the bundled
map (pinned with the library version in ``uv.lock``) is forced: costs are reproducible
and imports work offline.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from docseek.core.config import ModelsConfig, Settings, load_models_config
from docseek.core.interfaces import LLMClient
from docseek.core.logging import get_logger
from docseek.core.models import LLMResponse, ToolCall
from docseek.core.tracing import Tracer
from docseek.llm.cache import SQLiteLLMCache, cache_key
from docseek.llm.cost import BudgetGuard, SpendLedger

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

log = get_logger(__name__)

CompletionFn = Callable[..., Any]
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(?P<body>.*)\n```\s*$", re.S)


class StructuredOutputError(ValueError):
    """The model's reply does not validate against the requested ``response_format``."""


def parse_structured(content: str | None, response_format: type[BaseModel]) -> dict[str, Any]:
    """Validate a JSON reply (optionally wrapped in a ```json fence) into a plain dict."""
    if content is None:
        raise StructuredOutputError("empty reply")
    m = _JSON_FENCE_RE.match(content)
    raw = m["body"] if m else content
    try:
        return response_format.model_validate(json.loads(raw)).model_dump(mode="json")
    except (json.JSONDecodeError, ValidationError) as exc:
        raise StructuredOutputError(f"{response_format.__name__}: {exc}") from exc


class LiteLLMClient:
    """Provider-agnostic completions via LiteLLM.

    Args:
        api_keys: Key per provider prefix (``"gemini"``, ``"openai"``); passed per call, so
            secrets never have to be copied into ``os.environ``.
        model_params: Extra LiteLLM kwargs per model (e.g. ``reasoning_effort``).
        rpm_limits: Requests-per-minute cap per model, enforced client-side.
        num_retries: LiteLLM retries (exponential backoff) on transient errors.
        timeout_s: Per-request timeout.
        completion: Injected completion function (tests); defaults to ``litellm.completion``.
    """

    def __init__(
        self,
        api_keys: dict[str, str] | None = None,
        *,
        model_params: Mapping[str, Mapping[str, Any]] | None = None,
        rpm_limits: Mapping[str, int] | None = None,
        num_retries: int = 3,
        timeout_s: float = 120.0,
        completion: CompletionFn | None = None,
    ) -> None:
        self._api_keys = api_keys or {}
        self._model_params = model_params or {}
        self._limiters = {m: RateLimiter(rpm) for m, rpm in (rpm_limits or {}).items()}
        self._num_retries = num_retries
        self._timeout_s = timeout_s
        self._completion = completion

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if model is None:
            raise ValueError("LiteLLMClient needs an explicit model (from configs/models.yaml)")
        import litellm

        completion = self._completion or litellm.completion
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "num_retries": self._num_retries,
            "timeout": self._timeout_s,
        }
        if tools:
            kwargs["tools"] = tools
        if response_format is not None:
            kwargs["response_format"] = response_format
        kwargs.update(self._model_params.get(model, {}))
        provider = model.split("/", 1)[0]
        if provider in self._api_keys:
            kwargs["api_key"] = self._api_keys[provider]

        if model in self._limiters:
            self._limiters[model].wait()
        start = time.perf_counter()
        resp = completion(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        message = resp.choices[0].message
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (message.tool_calls or [])
        ]
        usage = getattr(resp, "usage", None)
        parsed = None
        if response_format is not None:
            try:
                parsed = parse_structured(message.content, response_format)
            except StructuredOutputError as exc:
                # Not raised here: the call was paid for, and the caller must account for it.
                log.warning("llm.structured_output_invalid", model=model, error=str(exc))
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            parsed=parsed,
            model=model,
            tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
            cost_usd=_cost(resp, model),
            latency_ms=latency_ms,
        )


class RateLimiter:
    """Spaces calls at least ``60 / rpm`` seconds apart (shared by all threads)."""

    def __init__(
        self,
        rpm: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        self.interval = 60.0 / rpm
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            self._sleep(start - now)


LOCAL_PROVIDERS = ("ollama", "ollama_chat")


def _cost(resp: Any, model: str) -> float:
    import litellm

    if model.split("/", 1)[0] in LOCAL_PROVIDERS:
        return 0.0  # runs on the local GPU
    try:
        return float(litellm.completion_cost(completion_response=resp, model=model))
    except Exception as exc:  # unknown price must not crash a run; it is logged loudly
        log.warning("llm.cost_unknown", model=model, error=str(exc))
        return 0.0


class CachedLLMClient:
    """Cache + budget + trace around an ``LLMClient``.

    A cache hit costs nothing and is traced with ``cost_usd=0`` and ``cached=True`` (its
    original price is kept as ``output.original_cost_usd``). Misses check the budget first,
    then call the inner client, record the spend and store the response. Replies that fail
    structured validation are paid for and traced, then raise ``StructuredOutputError``
    without being cached.
    """

    def __init__(
        self,
        inner: LLMClient,
        cache: SQLiteLLMCache,
        budget: BudgetGuard,
        tracer: Tracer | None = None,
        *,
        ledger: SpendLedger | None = None,
        spend_limits: Mapping[str, float] | None = None,
        model_params: Mapping[str, Mapping[str, Any]] | None = None,
        call_name: str = "llm",
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.budget = budget
        self.tracer = tracer or Tracer()
        self.ledger = ledger
        self.spend_limits = dict(spend_limits or {})
        # Must match what the inner client sends: they shape the answer, so they key the cache.
        self.model_params = model_params or {}
        self.call_name = call_name

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: type[BaseModel] | None = None,
        temperature: float = 0.0,
        name: str | None = None,
    ) -> LLMResponse:
        """Like ``LLMClient.complete``; ``name`` labels the trace event (e.g. ``synth.query``)."""
        if model is None:
            raise ValueError("model is required (read it from configs/models.yaml)")
        key = cache_key(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            response_format=response_format,
            extra=self.model_params.get(model),
        )
        start = time.perf_counter()
        hit = self.cache.get(key)
        if hit is not None:
            response = hit.model_copy(update={"cached": True})
            if response_format is not None and response.parsed is None:
                response = response.model_copy(
                    update={"parsed": parse_structured(response.content, response_format)}
                )
            self._trace(name, key, model, response, (time.perf_counter() - start) * 1000, 0.0)
            return response

        self.budget.check()
        if self.ledger is not None:
            self.ledger.check(model, self.spend_limits)
        response = self.inner.complete(
            messages,
            model=model,
            tools=tools,
            response_format=response_format,
            temperature=temperature,
        )
        self.budget.add(response.cost_usd)
        if self.ledger is not None:
            self.ledger.record(
                model=model,
                name=name or self.call_name,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                cost_usd=response.cost_usd,
            )
        self._trace(name, key, model, response, response.latency_ms, response.cost_usd)
        if response_format is not None and response.parsed is None:
            # Paid and traced above, but never cached: a retry should get a fresh answer.
            raise StructuredOutputError(
                f"{model} reply does not match {response_format.__name__}: "
                f"{(response.content or '')[:200]!r}"
            )
        self.cache.put(key, response)
        return response

    def _trace(
        self,
        name: str | None,
        key: str,
        model: str,
        response: LLMResponse,
        latency_ms: float,
        spent: float,
    ) -> None:
        self.tracer.record(
            kind="llm_call",
            name=name or self.call_name,
            input={"model": model, "cache_key": key},
            output={
                "cached": response.cached,
                "original_cost_usd": response.cost_usd,
                "n_tool_calls": len(response.tool_calls),
            },
            latency_ms=latency_ms,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            cost_usd=spent,
        )


def build_llm_client(
    settings: Settings,
    tracer: Tracer | None = None,
    *,
    models: ModelsConfig | None = None,
    completion: CompletionFn | None = None,
) -> CachedLLMClient:
    """The production stack: LiteLLM -> SQLite cache -> run budget + project ledger -> tracer.

    Per-model parameters, rpm caps and project spend limits come from ``configs/models.yaml``.
    """
    models = models or load_models_config(settings.configs_dir)
    keys = {
        provider: secret.get_secret_value()
        for provider, secret in (
            ("gemini", settings.gemini_api_key),
            ("openai", settings.openai_api_key),
        )
        if secret is not None and secret.get_secret_value().strip()
    }
    params = models.model_params()
    return CachedLLMClient(
        LiteLLMClient(
            keys, model_params=params, rpm_limits=models.rpm_limits(), completion=completion
        ),
        SQLiteLLMCache(settings.llm_cache_path),
        BudgetGuard(settings.budget_usd),
        tracer,
        ledger=SpendLedger(settings.llm_cache_path),
        spend_limits=models.spend_limits_usd,
        model_params=params,
    )


def is_degradable(exc: BaseException) -> bool:
    """True for LLM failures the pipeline may degrade around (bad JSON, provider/API errors).

    Budget and spend-limit stops are never degradable: they must halt the run. With
    ``DOCSEEK_LLM_STRICT=1`` (experiments) only invalid structured output degrades, so a quota
    or provider outage fails the run instead of silently scoring the un-expanded pipeline.
    """
    import os

    from docseek.llm.cost import BudgetExceededError

    if isinstance(exc, BudgetExceededError) or not isinstance(exc, Exception):
        return False
    if os.environ.get("DOCSEEK_LLM_STRICT") == "1":
        return isinstance(exc, StructuredOutputError)
    return True
