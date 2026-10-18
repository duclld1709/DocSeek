"""Fetch tests against a throwaway local git repo cloned over file:// (no network)."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docseek.cli import app
from docseek.core.config import FastAPICorpusConfig
from docseek.data.fetch import FetchError, Manifest, fetch_fastapi_docs

FILES_V1 = {
    "README.md": b"# root file outside sparse paths\n",
    "other/notes.md": b"not fetched\n",
    "docs/en/docs/index.md": b"# FastAPI { #fastapi }\n\nHello\n",
    "docs/en/docs/tutorial/first-steps.md": b"# First Steps { #first-steps }\n",
    "docs/es/docs/index.md": b"# Spanish, outside sparse paths\n",
    "docs_src/first_steps/tutorial001_py310.py": b"from fastapi import FastAPI\n",
}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> tuple[Path, str, str]:
    """A repo with two commits; returns (path, first_sha, second_sha)."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "master")
    for rel, content in FILES_V1.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v1")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "docs/en/docs/index.md").write_bytes(b"# FastAPI { #fastapi }\n\nChanged\n")
    (repo / "docs_src/new.py").write_bytes(b"x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v2")
    return repo, first, _git(repo, "rev-parse", "HEAD")


def _cfg(repo: Path, commit: str, dest: Path) -> FastAPICorpusConfig:
    return FastAPICorpusConfig(
        source="fastapi-docs",
        repo_url=repo.as_uri(),
        commit=commit,
        dest=dest,
        sparse_paths=["docs/en/docs", "docs_src"],
        docs_root="docs/en/docs",
        code_root="docs_src",
    )


def test_fetch_checks_out_pinned_commit_and_writes_manifest(
    upstream: tuple[Path, str, str], tmp_path: Path
) -> None:
    repo, first, _ = upstream
    dest, manifest_path = tmp_path / "raw/fastapi", tmp_path / "raw/MANIFEST.json"

    result = fetch_fastapi_docs(_cfg(repo, first, dest), manifest_path)

    assert result.cloned
    m = result.manifest
    assert m.commit == first
    assert [f.path for f in m.files] == [
        "docs/en/docs/index.md",
        "docs/en/docs/tutorial/first-steps.md",
        "docs_src/first_steps/tutorial001_py310.py",
    ]
    assert m.counts_by_extension == {".md": 2, ".py": 1}
    assert m.counts_by_path == {"docs/en/docs": 2, "docs_src": 1}
    index = next(f for f in m.files if f.path == "docs/en/docs/index.md")
    assert index.sha256 == hashlib.sha256(FILES_V1["docs/en/docs/index.md"]).hexdigest()
    assert (dest / "docs/en/docs/index.md").read_bytes() == FILES_V1["docs/en/docs/index.md"]
    assert not (dest / "docs/es").exists()
    assert not (dest / "other").exists()

    on_disk = Manifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    assert on_disk == m


def test_fetch_keeps_history_for_updated_at(
    upstream: tuple[Path, str, str], tmp_path: Path
) -> None:
    repo, _, second = upstream
    dest = tmp_path / "raw/fastapi"
    fetch_fastapi_docs(_cfg(repo, second, dest), tmp_path / "m.json")

    log = _git(dest, "log", "--format=%s", "--", "docs/en/docs/index.md")
    assert log.splitlines() == ["v2", "v1"]


def test_fetch_is_idempotent_and_force_reclones(
    upstream: tuple[Path, str, str], tmp_path: Path
) -> None:
    repo, first, _ = upstream
    cfg = _cfg(repo, first, tmp_path / "raw/fastapi")
    first_run = fetch_fastapi_docs(cfg, tmp_path / "m.json")

    again = fetch_fastapi_docs(cfg, tmp_path / "m.json")
    forced = fetch_fastapi_docs(cfg, tmp_path / "m.json", force=True)

    assert not again.cloned
    assert forced.cloned
    assert again.manifest.files == first_run.manifest.files == forced.manifest.files


def test_fetch_moves_existing_clone_to_new_pin(
    upstream: tuple[Path, str, str], tmp_path: Path
) -> None:
    repo, first, second = upstream
    dest = tmp_path / "raw/fastapi"
    fetch_fastapi_docs(_cfg(repo, first, dest), tmp_path / "m.json")

    result = fetch_fastapi_docs(_cfg(repo, second, dest), tmp_path / "m.json")

    assert result.cloned
    assert result.manifest.commit == second
    assert "docs_src/new.py" in {f.path for f in result.manifest.files}


def test_fetch_unknown_commit_fails(upstream: tuple[Path, str, str], tmp_path: Path) -> None:
    repo, _, _ = upstream
    with pytest.raises(FetchError):
        fetch_fastapi_docs(_cfg(repo, "0" * 40, tmp_path / "raw/fastapi"), tmp_path / "m.json")


def test_cli_data_fetch(
    upstream: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, first, _ = upstream
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "corpus.yaml").write_text(
        f"""
manifest_path: {(tmp_path / "MANIFEST.json").as_posix()}
fastapi:
  source: fastapi-docs
  repo_url: {repo.as_uri()}
  commit: {first}
  dest: {(tmp_path / "fastapi").as_posix()}
  sparse_paths: [docs/en/docs, docs_src]
  docs_root: docs/en/docs
  code_root: docs_src
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCSEEK_CONFIGS_DIR", str(configs))

    result = CliRunner().invoke(app, ["data", "fetch"])

    assert result.exit_code == 0, result.output
    assert "3 files" in result.output
    assert (tmp_path / "MANIFEST.json").exists()
