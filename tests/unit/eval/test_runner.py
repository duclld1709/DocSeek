"""Experiment runner and comparison on the fixture corpora: Oracle = 1, Random < Oracle."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from docseek.cli import app
from docseek.core.acl import AccessPolicy
from docseek.core.config import load_principals
from docseek.core.tracing import read_traces
from docseek.eval.golden import load_golden
from docseek.eval.runner import (
    ExperimentConfig,
    RetrieverConfig,
    build_retriever,
    compare_runs,
    load_per_query,
    run_experiment,
)
from docseek.ingest.pipeline import write_jsonl
from docseek.testing.fixtures import MINI_GOLDEN, REPO_ROOT, mini_ingest

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))
GOLDEN = load_golden(MINI_GOLDEN)
RESULT = mini_ingest()


def _run(tmp_path: Path, retriever: str, split: str = "all") -> Path:
    cfg = ExperimentConfig(name=retriever, retriever=RetrieverConfig(type=retriever))
    return run_experiment(
        cfg,
        golden=GOLDEN,
        sections=RESULT.sections,
        chunks=RESULT.chunks,
        policy=POLICY,
        split=split,  # type: ignore[arg-type]
        experiments_dir=tmp_path,
        golden_path=MINI_GOLDEN,
    )


def _metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def test_oracle_scores_perfectly_without_leaks(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "oracle")
    m = _metrics(run_dir)
    overall = m["overall"]
    assert overall["n_queries"] == len(GOLDEN)
    for name in ("recall@10", "ndcg@10", "mrr@1", "map@10", "hit@1"):
        assert overall[name]["mean"] == pytest.approx(1.0), name
        assert overall[name]["ci_low"] == pytest.approx(1.0)
    n_answerable = sum(1 for q in GOLDEN if q.relevant)
    assert overall["recall@10"]["n"] == n_answerable
    assert overall["leakage@10"] == {"total": 0, "max": 0}
    assert overall["stale_hit@1"]["mean"] == 0.0  # q0009: fresh doc ranked first
    assert set(m["by_qtype"]) >= {"howto", "freshness", "permission", "unanswerable"}


def test_run_directory_contents(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "oracle")
    assert run_dir.name.endswith("_oracle_all")
    rows = load_per_query(run_dir)
    assert set(rows) == {q.qid for q in GOLDEN}
    q1 = rows["q0001"]
    assert all(rank is not None for rank in q1["relevant_ranks"].values())
    assert "leakage@10" in rows["q0012"]["metrics"] and "ndcg@10" not in rows["q0012"]["metrics"]
    events = read_traces(run_dir / "traces.jsonl")
    assert len(events) == len(GOLDEN) and {e.kind for e in events} == {"retrieve"}
    assert len({e.trace_id for e in events}) == len(GOLDEN)
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    assert env["seed"] == 13 and len(env["config_hash"]) == 64 and len(env["golden_sha256"]) == 64


def test_random_is_worse_than_oracle_and_never_leaks(tmp_path: Path) -> None:
    oracle = _metrics(_run(tmp_path, "oracle"))["overall"]
    random_ = _metrics(_run(tmp_path, "random"))["overall"]
    assert random_["leakage@10"]["total"] == 0
    assert random_["ndcg@10"]["mean"] < oracle["ndcg@10"]["mean"]
    assert random_["mrr@1"]["mean"] < 0.5


def test_split_filter(tmp_path: Path) -> None:
    golden = [q.model_copy(update={"split": "test" if q.qid < "q0005" else "dev"}) for q in GOLDEN]
    cfg = ExperimentConfig(name="oracle", retriever=RetrieverConfig(type="oracle"))
    run_dir = run_experiment(
        cfg,
        golden=golden,
        sections=RESULT.sections,
        chunks=RESULT.chunks,
        policy=POLICY,
        split="test",
        experiments_dir=tmp_path,
    )
    assert set(load_per_query(run_dir)) == {"q0001", "q0002", "q0003", "q0004"}
    with pytest.raises(ValueError, match="no golden queries"):
        run_experiment(
            cfg,
            golden=[q.model_copy(update={"split": "dev"}) for q in GOLDEN],
            sections=RESULT.sections,
            chunks=RESULT.chunks,
            policy=POLICY,
            split="test",
            experiments_dir=tmp_path,
        )


def test_unknown_retriever_type() -> None:
    cfg = ExperimentConfig(name="x", retriever=RetrieverConfig(type="no-such-retriever"))
    with pytest.raises(ValueError, match="unknown retriever type"):
        build_retriever(cfg, RESULT.chunks, GOLDEN, POLICY)


def test_compare_runs_detects_the_oracle_advantage(tmp_path: Path) -> None:
    a = _run(tmp_path, "random")
    b = _run(tmp_path, "oracle")
    rows, swings = compare_runs(a, b, metrics=["ndcg@10", "recall@10"])
    by_name = {r.metric: r for r in rows}
    assert by_name["ndcg@10"].delta > 0 and by_name["ndcg@10"].n == 12
    assert by_name["ndcg@10"].verdict == "better" and by_name["ndcg@10"].p_value < 0.05
    assert swings[-1][2] >= swings[0][2]
    same, _ = compare_runs(b, b, metrics=["ndcg@10"])
    assert same[0].verdict == "no significant difference" and same[0].p_value == 1.0


def test_cli_eval_run_and_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed = tmp_path / "processed"
    write_jsonl(processed / "sections.jsonl", RESULT.sections)
    write_jsonl(processed / "chunks.jsonl", RESULT.chunks)
    configs = {}
    for name in ("oracle", "random"):
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": name,
                    "retriever": {"type": name},
                    "golden": MINI_GOLDEN.as_posix(),
                    "processed_dir": processed.as_posix(),
                }
            ),
            encoding="utf-8",
        )
        configs[name] = path
    monkeypatch.setenv("DOCSEEK_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    runner = CliRunner()
    for path in configs.values():
        result = runner.invoke(app, ["eval", "run", "--config", str(path), "--split", "all"])
        assert result.exit_code == 0, result.output
        assert "recall@10" in result.output
    runs = sorted((tmp_path / "experiments").iterdir())
    assert len(runs) == 2
    result = runner.invoke(app, ["eval", "compare", str(runs[1]), str(runs[0])])
    assert result.exit_code == 0, result.output
    assert "ndcg@10" in result.output
    bad = runner.invoke(app, ["eval", "run", "--config", str(configs["oracle"]), "--split", "x"])
    assert bad.exit_code == 1
