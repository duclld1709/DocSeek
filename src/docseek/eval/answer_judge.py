"""DeepEval answer metrics on our own LLM client (cache + BudgetGuard + spend ledger).

DeepEval normally calls OpenAI itself; :class:`DocSeekJudge` routes every judge prompt
through ``CachedLLMClient`` instead, so judge calls are cached, budgeted, traced and counted
in the project ledger like every other LLM call (CLAUDE.md). The judge model comes from the
``judge`` role (a different provider family than the answer model).

Metrics per answered question: Faithfulness, AnswerRelevancy, ContextualRecall (needs the
reference answer) and a GEval "citation support" score (are cited sections what supports the
claims). Reasons are switched off to save judge calls.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from docseek.llm.client import CachedLLMClient

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from deepeval.models import DeepEvalBaseLLM

CITATION_CRITERIA = (
    "The actual output cites sources as [section_id]. The retrieval context lists each cited "
    "section as 'section_id: text'. Judge whether every factual claim in the actual output is "
    "supported by the text of the section(s) cited next to it. Penalise claims whose cited "
    "section does not contain them, and claims without any citation."
)


class DocSeekJudge(DeepEvalBaseLLM):
    def __init__(self, llm: CachedLLMClient, model: str) -> None:
        self._llm = llm
        self._model_name = model
        super().__init__(model)  # type: ignore[no-untyped-call]

    def load_model(self, *args: Any, **kwargs: Any) -> DocSeekJudge:
        return self

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        resp = self._llm.complete(
            [{"role": "user", "content": prompt}],
            model=self._model_name,
            response_format=schema,
            name="deepeval",
        )
        if schema is not None:
            return schema.model_validate(resp.parsed)
        return resp.content or ""

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any
    ) -> Any:
        return self.generate(prompt, schema, **kwargs)

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return str(self._model_name)

    def supports_structured_outputs(self) -> bool:
        return True


def build_metrics(judge: DocSeekJudge) -> dict[str, Any]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import SingleTurnParams

    common: dict[str, Any] = {"model": judge, "include_reason": False, "async_mode": False}
    return {
        "faithfulness": FaithfulnessMetric(**common),
        "answer_relevancy": AnswerRelevancyMetric(**common),
        "contextual_recall": ContextualRecallMetric(**common),
        "citation_support": GEval(
            name="Citation support",
            criteria=CITATION_CRITERIA,
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
            model=judge,
            async_mode=False,
        ),
    }


def score_answer(
    metrics: dict[str, Any],
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str | None,
) -> dict[str, float | None]:
    """Scores in [0, 1]; ``contextual_recall`` is ``None`` without a reference answer."""
    from deepeval.test_case import LLMTestCase

    case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=reference,
        retrieval_context=list(contexts) or ["(no context)"],
    )
    out: dict[str, float | None] = {}
    for name, metric in metrics.items():
        if name == "contextual_recall" and not reference:
            out[name] = None
            continue
        metric.measure(case)
        out[name] = float(metric.score) if metric.score is not None else None
    return out
