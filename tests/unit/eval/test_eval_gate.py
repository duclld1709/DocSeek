"""Regression gate (PROJECT_SPEC §8): fake-friendly configs on the fixture corpora must not
score below tests/fixtures/baseline_metrics.json, and Leakage must be 0.

Refresh the baseline deliberately with ``DOCSEEK_UPDATE_BASELINE=1 uv run pytest <this file>``.
"""

import json
import os
from pathlib import Path

import pytest

from docseek.core.acl import AccessPolicy
from docseek.core.config import load_principals
from docseek.eval.golden import load_golden
from docseek.eval.runner import ExperimentConfig, RetrieverConfig, run_experiment
from docseek.testing.fixtures import FIXTURES, MINI_GOLDEN, REPO_ROOT, mini_ingest

BASELINE = FIXTURES / "baseline_metrics.json"
GATED = ("ndcg@10", "recall@10", "mrr@10")
TOLERANCE = 1e-6

CONFIGS: dict[str, dict] = {
    "bm25f": {
        "type": "bm25",
        "k1": 1.5,
        "b": 0.5,
        "fields": {"title": 2.0, "section_path": 0.5, "body": 1.0, "code": 0.5},
        "index_dir": None,
    },
    "tfidf": {"type": "tfidf", "index_dir": None},
}


def _run(name: str, tmp_path: Path) -> dict:
    result = mini_ingest()
    cfg = ExperimentConfig(name=name, retriever=RetrieverConfig(**CONFIGS[name]))
    run_dir = run_experiment(
        cfg,
        golden=load_golden(MINI_GOLDEN),
        sections=result.sections,
        chunks=result.chunks,
        policy=AccessPolicy(load_principals(REPO_ROOT / "configs")),
        split="all",
        experiments_dir=tmp_path,
    )
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["overall"]


def test_gate(tmp_path: Path) -> None:
    current = {name: _run(name, tmp_path) for name in CONFIGS}
    for name, overall in current.items():
        assert overall["leakage@10"]["total"] == 0, name
    observed = {n: {m: o[m]["mean"] for m in GATED} for n, o in current.items()}
    if os.environ.get("DOCSEEK_UPDATE_BASELINE"):
        BASELINE.write_text(json.dumps(observed, indent=2) + "\n", encoding="utf-8", newline="\n")
        pytest.skip("baseline updated")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    for name, metrics in baseline.items():
        assert name in observed, f"gated config {name} no longer runs"
        for m, value in metrics.items():
            assert observed[name][m] >= value - TOLERANCE, (name, m, observed[name][m], value)
