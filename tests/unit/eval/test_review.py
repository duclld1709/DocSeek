"""Human-in-the-loop helpers and their interactive CLI commands (scripted input)."""

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from docseek.cli import app
from docseek.core.acl import AccessPolicy
from docseek.core.config import load_principals
from docseek.core.models import QueryType
from docseek.eval.golden import GoldenValidationError, load_golden, write_golden
from docseek.eval.judge import Judged
from docseek.eval.review import (
    CalibrationPair,
    HumanLabel,
    HumanQuery,
    ReviewDecision,
    apply_decisions,
    build_golden,
    calibration_report,
    find_sections,
    latest_decisions,
    load_human_queries,
    read_jsonl_models,
    review_order,
    sample_calibration_pairs,
)
from docseek.ingest.pipeline import write_jsonl
from docseek.testing.fixtures import MINI_GOLDEN, REPO_ROOT, mini_ingest

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))
SECTIONS = mini_ingest().sections
BY_ID = {s.section_id: s for s in SECTIONS}
GOLDEN = load_golden(MINI_GOLDEN)


def test_review_order_round_robins_qtypes() -> None:
    order = review_order(GOLDEN)
    assert sorted(q.qid for q in order) == sorted(q.qid for q in GOLDEN)
    n_types = len({q.qtype for q in GOLDEN})
    assert len({q.qtype for q in order[:n_types]}) == n_types


def test_apply_decisions_and_last_decision_wins() -> None:
    decisions = latest_decisions(
        [
            ReviewDecision(qid="q0001", action="drop"),
            ReviewDecision(qid="q0001", action="accept"),
            ReviewDecision(qid="q0002", action="edit", text="BackgroundTasks add_task"),
            ReviewDecision(qid="q0003", action="drop", note="ambiguous"),
        ]
    )
    out = {q.qid: q for q in apply_decisions(GOLDEN, decisions)}
    assert out["q0001"].reviewed and "q0003" not in out
    assert out["q0002"].text == "BackgroundTasks add_task" and out["q0002"].reviewed
    assert out["q0004"] == next(q for q in GOLDEN if q.qid == "q0004")  # untouched


def test_build_golden_merges_renumbers_and_validates() -> None:
    human = [
        HumanQuery(
            text="what is the processor timeout",
            qtype=QueryType.FRESHNESS,
            principal="engineer",
            relevant={"internal-kb:kb-0003.md#payment-api-timeouts": 2},
            stale=["internal-kb:kb-0002.md#payment-api-timeouts"],
        )
    ]
    decisions = {"q0003": ReviewDecision(qid="q0003", action="drop")}
    golden = build_golden(GOLDEN, decisions, human, sections=SECTIONS, policy=POLICY)
    assert [q.qid for q in golden] == [f"q{i:04d}" for i in range(1, len(GOLDEN) + 1)]
    last = golden[-1]
    assert last.origin == "human" and last.reviewed and last.lexical_overlap is not None
    assert {q.split for q in golden} == {"dev", "test"}


def test_build_golden_rejects_hidden_relevant_sections() -> None:
    human = [
        HumanQuery(
            text="jwt rotation",
            qtype=QueryType.FACTOID,
            principal="engineer",
            relevant={"internal-kb:kb-0004.md#jwt-signing-key-rotation": 2},
        )
    ]
    with pytest.raises(GoldenValidationError, match="cannot read"):
        build_golden(GOLDEN, {}, human, sections=SECTIONS, policy=POLICY)


def test_load_human_queries(tmp_path: Path) -> None:
    path = tmp_path / "h.yaml"
    path.write_text(
        yaml.safe_dump({"queries": [{"text": "cors", "qtype": "howto", "relevant": {"a#b": 2}}]}),
        encoding="utf-8",
    )
    (q,) = load_human_queries(path)
    assert (q.text, q.qtype, q.principal) == ("cors", QueryType.HOWTO, "anonymous")
    assert load_human_queries(tmp_path / "missing.yaml") == []


def test_find_sections_puts_title_matches_first() -> None:
    found = find_sections(SECTIONS, "dependencies")
    assert found[0].title.lower().startswith("dependencies")
    assert all(
        "dependencies" in (s.title + " ".join(s.section_path) + s.text).lower() for s in found
    )


def test_calibration_pairs_mix_strata_and_respect_acl() -> None:
    pairs = sample_calibration_pairs(GOLDEN, SECTIONS, POLICY, n=12)
    assert len(pairs) <= 12 and [p.pair_id for p in pairs] == [
        f"p{i:03d}" for i in range(1, len(pairs) + 1)
    ]
    assert {p.stratum for p in pairs} >= {"relevant", "same_doc_type"}
    principal = {q.qid: q.principal for q in GOLDEN}
    for p in pairs:
        assert POLICY.can_read(principal[p.qid], BY_ID[p.section_id].acl)
    assert pairs == sample_calibration_pairs(GOLDEN, SECTIONS, POLICY, n=12)


def _judged(score: int) -> Judged:
    return Judged(score=score, reason="r", rubric_version="abc", model="openai/j")


def test_calibration_report() -> None:
    pairs = [
        CalibrationPair(
            pair_id=f"p{i:03d}",
            qid="q0001",
            query="q",
            section_id=SECTIONS[i].section_id,
            stratum="relevant" if i < 3 else "same_doc_type",
        )
        for i in range(6)
    ]
    human = {
        p.pair_id: HumanLabel(pair_id=p.pair_id, score=s)
        for p, s in zip(pairs, [2, 2, 1, 0, 0, 1], strict=True)
    }
    judge_qf = {p.pair_id: _judged(s) for p, s in zip(pairs, [2, 1, 1, 0, 0, 2], strict=True)}
    judge_sf = {p.pair_id: _judged(s) for p, s in zip(pairs, [2, 2, 1, 0, 0, 2], strict=True)}
    report = calibration_report(pairs, human, judge_qf, judge_sf, BY_ID)
    assert report["n_pairs"] == 6 and report["exact_agreement"] == pytest.approx(4 / 6)
    assert report["confusion_human_rows_judge_cols"] == [[2, 0, 0], [0, 1, 1], [0, 1, 1]]
    assert -1 <= report["kappa_linear"] <= 1 and report["rubric_version"] == "abc"
    assert len(report["disagreements"]) == 2
    assert report["position_bias"]["agreement"] == pytest.approx(5 / 6)
    assert set(report["by_stratum"]) == {"relevant", "same_doc_type"}


# --- CLI ----------------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    configs = tmp_path / "configs"
    shutil.copytree(REPO_ROOT / "configs", configs)
    processed = tmp_path / "processed"
    write_jsonl(processed / "sections.jsonl", SECTIONS)
    (configs / "ingest.yaml").write_text(
        yaml.safe_dump({"output_dir": processed.as_posix(), "chunking": {}}), encoding="utf-8"
    )
    monkeypatch.setenv("DOCSEEK_CONFIGS_DIR", str(configs))
    monkeypatch.setenv("DOCSEEK_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    write_golden(tmp_path / "draft.jsonl", GOLDEN)
    return tmp_path


def test_cli_review_records_decisions_and_resumes(workspace: Path) -> None:
    runner = CliRunner()
    args = [
        "eval",
        "review",
        "--draft",
        str(workspace / "draft.jsonl"),
        "--decisions",
        str(workspace / "rev.jsonl"),
    ]
    first = runner.invoke(app, args, input="a\nd\nnot useful\ne\nnew text\n\nq\n")
    assert first.exit_code == 0, first.output
    decisions = read_jsonl_models(workspace / "rev.jsonl", ReviewDecision)
    assert [d.action for d in decisions] == ["accept", "drop", "edit"]
    assert decisions[1].note == "not useful" and decisions[2].text == "new text"
    second = runner.invoke(app, args, input="s\nq\n")
    assert "3 reviewed so far" in " ".join(second.output.split())


def test_cli_build_label_and_calibrate(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    out = workspace / "golden.jsonl"
    built = runner.invoke(
        app,
        [
            "eval",
            "build-golden",
            "--draft",
            str(workspace / "draft.jsonl"),
            "--decisions",
            str(workspace / "none.jsonl"),
            "--human",
            str(workspace / "none.yaml"),
            "--out",
            str(out),
        ],
    )
    assert built.exit_code == 0, built.output
    assert len(load_golden(out)) == len(GOLDEN)

    pairs, labels = workspace / "pairs.jsonl", workspace / "labels.jsonl"
    label_args = [
        "eval",
        "label",
        "--golden",
        str(out),
        "--pairs",
        str(pairs),
        "--labels",
        str(labels),
        "--n",
        "6",
    ]
    labelled = runner.invoke(app, label_args, input="2\n\n0\nbad match\n7\n1\n\nq\n")
    assert labelled.exit_code == 0, labelled.output
    got = read_jsonl_models(labels, HumanLabel)
    assert [lab.score for lab in got] == [2, 0, 1] and got[1].note == "bad match"

    from docseek.core.tracing import MemoryTraceSink, Tracer
    from docseek.llm import client as llm_client
    from docseek.llm.cache import SQLiteLLMCache
    from docseek.llm.cost import BudgetGuard
    from docseek.testing.fakes import ScriptedLLM

    reply = '{"score": 1, "reason": "r"}'
    fake = llm_client.CachedLLMClient(
        ScriptedLLM([reply] * 20),
        SQLiteLLMCache(Path(":memory:")),
        BudgetGuard(1),
        Tracer(MemoryTraceSink()),
    )
    monkeypatch.setattr(llm_client, "build_llm_client", lambda *a, **k: fake)
    calibrated = runner.invoke(
        app, ["eval", "calibrate", "--pairs", str(pairs), "--labels", str(labels)]
    )
    assert calibrated.exit_code == 0, calibrated.output
    assert "kappa linear=" in calibrated.output
    (run,) = (workspace / "experiments").iterdir()
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report["n_pairs"] == 3 and report["judge_label_counts"] == {"1": 3}
