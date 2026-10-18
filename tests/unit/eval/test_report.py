"""Golden statistics, the overlap SVG and per-run cost summaries."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docseek.cli import app
from docseek.core.tracing import JsonlTraceSink, Tracer
from docseek.eval.golden import load_golden
from docseek.eval.report import golden_stats, overlap_bins, overlap_svg, run_costs
from docseek.testing.fixtures import MINI_GOLDEN

GOLDEN = [
    q.model_copy(update={"style": s, "lexical_overlap": o})
    for q, s, o in zip(
        load_golden(MINI_GOLDEN),
        ["human", "machine", "paraphrase"] * 5,
        [0.0, 0.25, 0.55, 1.0, 0.95, 0.1] * 3,
        strict=False,
    )
]


def test_golden_stats() -> None:
    stats = golden_stats(GOLDEN)
    assert stats["n_queries"] == 14 and stats["reviewed"] == 14
    assert stats["by_qtype"]["howto"] == 4
    assert stats["relevant_by_source"]["internal-kb"] >= 3
    assert set(stats["lexical_overlap_mean_by_style"]) == {"human", "machine", "paraphrase"}


def test_overlap_bins_and_svg_are_consistent() -> None:
    bins = overlap_bins(GOLDEN)
    assert sum(sum(v) for v in bins.values()) == 14
    assert bins["human"][0] >= 1 and bins["human"][9] >= 1  # 0.0 and 1.0 land in end bins
    svg = overlap_svg(GOLDEN, "Overlap <test>")
    root = ET.fromstring(svg)  # well-formed XML
    bars = [el for el in root.iter("{http://www.w3.org/2000/svg}path")]
    assert len(bars) == sum(1 for v in bins.values() for c in v if c)
    assert "prefers-color-scheme: dark" in svg and "Overlap &lt;test&gt;" in svg


def test_run_costs_split_paid_and_cached(tmp_path: Path) -> None:
    run = tmp_path / "20261018T000000Z_synth"
    tracer = Tracer(JsonlTraceSink(run / "traces.jsonl"))
    for model, cached, cost in [
        ("openai/n", False, 0.01),
        ("openai/n", True, 0.0),
        ("gemini/l", False, 0.02),
    ]:
        tracer.record(
            kind="llm_call",
            name="x",
            input={"model": model},
            output={"cached": cached},
            latency_ms=1,
            tokens_in=10,
            cost_usd=cost,
        )
    tracer.record(kind="retrieve", name="r", input={}, output={}, latency_ms=1)
    (row,) = run_costs(tmp_path)
    assert row["providers"]["openai"]["paid_calls"] == 1
    assert row["providers"]["openai"]["cached_calls"] == 1
    assert row["providers"]["openai"]["cost_usd"] == pytest.approx(0.01)
    assert row["providers"]["gemini"]["tokens_in"] == 10


def test_cli_golden_report_writes_svg_and_json(tmp_path: Path) -> None:
    svg, out = tmp_path / "img" / "o.svg", tmp_path / "s.json"
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "golden-report",
            "--golden",
            str(MINI_GOLDEN),
            "--svg",
            str(svg),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert svg.read_text(encoding="utf-8").startswith("<svg") and out.is_file()
