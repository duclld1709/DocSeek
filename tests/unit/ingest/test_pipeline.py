"""End-to-end ingest on the fixture corpora (mini_corpus + mini_kb), offline."""

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from docseek.cli import app
from docseek.core.config import CorpusConfig, load_corpus_config, load_principals
from docseek.core.models import Chunk, Section
from docseek.ingest.chunking import ChunkingConfig
from docseek.ingest.includes import find_unresolved_includes
from docseek.ingest.pipeline import (
    IngestResult,
    corpus_stats,
    read_jsonl,
    resolve_links,
    run_pipeline,
)
from docseek.ingest.tokens import RegexTokenCounter

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests" / "fixtures"
COUNTER = RegexTokenCounter()


def _corpus(kb_dir: Path = FIXTURES / "mini_kb") -> CorpusConfig:
    base = load_corpus_config(REPO / "configs")
    return base.model_copy(
        update={
            "fastapi": base.fastapi.model_copy(update={"dest": FIXTURES / "mini_corpus"}),
            "internal_kb": base.internal_kb.model_copy(update={"dir": kb_dir}),
        }
    )


@pytest.fixture(scope="module")
def result() -> IngestResult:
    return run_pipeline(
        _corpus(),
        load_principals(REPO / "configs"),
        COUNTER,
        ChunkingConfig(),
        repo_root=REPO,
        use_git_dates=False,
    )


def test_both_corpora_are_ingested_without_excluded_pages(result: IngestResult) -> None:
    docs = {s.doc_id for s in result.sections}
    assert "fastapi-docs:tutorial/first-steps.md" in docs
    assert "internal-kb:kb-0003.md" in docs
    assert not any("release-notes" in d or "fastapi-people" in d for d in docs)
    assert len({s.section_id for s in result.sections}) == len(result.sections)


def test_no_include_syntax_and_no_issues(result: IngestResult) -> None:
    assert result.issues == []
    assert all(find_unresolved_includes(s.text) == [] for s in result.sections)


def test_every_non_empty_section_is_chunked(result: IngestResult) -> None:
    chunked = {c.section_id for c in result.chunks}
    assert chunked == {s.section_id for s in result.sections if s.text.strip()}
    kb_chunk = next(c for c in result.chunks if c.section_id.startswith("internal-kb:kb-0004"))
    assert kb_chunk.metadata["acl"] == ["security-team"]


def test_dangling_link_is_dropped_and_reported(result: IngestResult) -> None:
    linked = {t for s in result.sections for t in s.links}
    assert not any("handling-errors" in t for t in linked)
    assert result.link_stats["dangling"] >= 1 and result.link_stats["ok"] >= 1


def test_resolve_links_falls_back_from_missing_anchor_to_document(result: IngestResult) -> None:
    first = result.sections[0].model_copy(
        update={"links": ["fastapi-docs:tutorial/path-params.md#no-such-anchor"]}
    )
    out, counts = resolve_links([first, *result.sections[1:]])
    assert out[0].links == ["fastapi-docs:tutorial/path-params.md"]
    assert counts["anchor_fallback"] >= 1


def test_stats_are_consistent(result: IngestResult) -> None:
    stats = corpus_stats(result, COUNTER)
    assert stats["sections"] == len(result.sections)
    assert stats["chunks"] == len(result.chunks)
    assert stats["by_source"]["internal-kb"]["documents"] == 4
    assert stats["by_acl"]["security-team"] == 1
    assert stats["by_status"]["deprecated"] == 1
    assert stats["unresolved_include_syntax"] == 0
    assert stats["token_counter"] == "regex"


def _configs(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    corpus = yaml.safe_load((configs / "corpus.yaml").read_text(encoding="utf-8"))
    corpus["fastapi"]["dest"] = (FIXTURES / "mini_corpus").as_posix()
    corpus["internal_kb"]["dir"] = (FIXTURES / "mini_kb").as_posix()
    corpus["manifest_path"] = (tmp_path / "MANIFEST.json").as_posix()
    (configs / "corpus.yaml").write_text(yaml.safe_dump(corpus), encoding="utf-8")
    ingest = {"output_dir": (tmp_path / "processed").as_posix(), "chunking": {"max_tokens": 512}}
    (configs / "ingest.yaml").write_text(yaml.safe_dump(ingest), encoding="utf-8")
    return configs


def test_cli_ingest_writes_outputs_and_skips_unchanged_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCSEEK_CONFIGS_DIR", str(_configs(tmp_path)))
    runner = CliRunner()
    first = runner.invoke(app, ["ingest", "--offline-tokens"])
    assert first.exit_code == 0, first.output
    out = tmp_path / "processed"
    sections = read_jsonl(out / "sections.jsonl", Section)
    chunks = read_jsonl(out / "chunks.jsonl", Chunk)
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["sections"] == len(sections) and stats["chunks"] == len(chunks)
    assert len(stats["input_hash"]) == 64

    second = runner.invoke(app, ["ingest", "--offline-tokens"])
    assert second.exit_code == 0 and "up to date" in " ".join(second.output.split())
    forced = runner.invoke(app, ["ingest", "--offline-tokens", "--force"])
    assert forced.exit_code == 0 and "up to date" not in " ".join(forced.output.split())
