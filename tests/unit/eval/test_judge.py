"""Relevance judge with a scripted LLM, plus the bias statistics."""

from pathlib import Path

import pytest

from docseek.core.config import LLMRoleConfig
from docseek.core.tracing import MemoryTraceSink, Tracer
from docseek.eval.judge import (
    RelevanceJudge,
    build_judge_messages,
    load_rubric,
    position_bias,
    render_section,
    rubric_version,
    spearman,
)
from docseek.llm.cache import SQLiteLLMCache
from docseek.llm.client import CachedLLMClient, StructuredOutputError
from docseek.llm.cost import BudgetGuard
from docseek.testing.fakes import ScriptedLLM
from docseek.testing.fixtures import mini_ingest

ROLE = LLMRoleConfig(model="openai/judge-model", temperature=0.0)
SECTIONS = {s.section_id: s for s in mini_ingest().sections}
YIELD = SECTIONS[
    "fastapi-docs:tutorial/dependencies/dependencies-with-yield.md#dependencies-with-yield"
]
OLD_KB = SECTIONS["internal-kb:kb-0002.md#payment-api-timeouts"]


def _judge(replies: list[str]) -> tuple[RelevanceJudge, ScriptedLLM]:
    inner = ScriptedLLM(replies)
    llm = CachedLLMClient(
        inner, SQLiteLLMCache(Path(":memory:")), BudgetGuard(1), Tracer(MemoryTraceSink())
    )
    return RelevanceJudge(llm, ROLE), inner


def test_rubric_defines_scale_examples_and_output() -> None:
    rubric = load_rubric()
    for needle in ("2 = answers directly", "1 = related or partial", "0 = not relevant"):
        assert needle in rubric
    assert '{"score": 0 | 1 | 2' in rubric and "data, not instructions" in rubric
    assert len(rubric_version(rubric)) == 12


def test_messages_wrap_untrusted_text_and_support_both_orders() -> None:
    q_first = build_judge_messages("cleanup dependency", YIELD, "RUBRIC")
    s_first = build_judge_messages("cleanup dependency", YIELD, "RUBRIC", order="section_first")
    assert q_first[0] == {"role": "system", "content": "RUBRIC"}
    assert q_first[1]["content"].index("<query>") < q_first[1]["content"].index("<section")
    assert s_first[1]["content"].index("<section") < s_first[1]["content"].index("<query>")
    assert "</section>" in q_first[1]["content"]


def test_render_section_marks_deprecated_and_truncates() -> None:
    assert 'status="deprecated"' in render_section(OLD_KB)
    long = YIELD.model_copy(update={"text": "x" * 10_000})
    rendered = render_section(long, max_chars=100)
    assert rendered.count("x") == 100 and "[...]" in rendered


def test_judge_returns_score_with_rubric_version_and_uses_judge_role() -> None:
    judge, inner = _judge(['{"score": 2, "reason": "explains yield cleanup"}'])
    out = judge.judge("cleanup dependency", YIELD)
    assert (out.score, out.reason, out.model) == (2, "explains yield cleanup", ROLE.model)
    assert out.rubric_version == judge.version
    assert inner.calls[0]["model"] == ROLE.model and inner.calls[0]["temperature"] == 0.0


def test_judge_caches_identical_pairs_and_rejects_bad_scores() -> None:
    judge, inner = _judge(['{"score": 1, "reason": "r"}', '{"score": 5, "reason": "r"}'])
    judge.judge("q", YIELD)
    judge.judge("q", YIELD)
    assert len(inner.calls) == 1
    with pytest.raises(StructuredOutputError):
        judge.judge("another query", YIELD)


def test_spearman() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3], [5, 5, 5]) == 0.0
    # ties get average ranks: x ranks [1.5, 1.5, 3], y ranks [1, 2, 3]
    assert spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(0.8660254, rel=1e-6)
    with pytest.raises(ValueError):
        spearman([1], [1])


def test_position_bias() -> None:
    pb = position_bias([2, 1, 0, 2], [2, 2, 0, 2])
    assert (pb.n, pb.agreement, pb.mean_shift) == (4, 0.75, 0.25)
    assert pb.kappa < 1
    assert position_bias([0, 1, 2], [0, 1, 2]).kappa == pytest.approx(1.0)
