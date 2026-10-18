"""doc_type mapping, exclusions and git-based updated_at (against a throwaway local repo)."""

import os
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from docseek.core.config import load_corpus_config
from docseek.ingest.metadata import doc_type_for, git_updated_at, is_excluded


@pytest.mark.parametrize(
    ("rel_path", "doc_type"),
    [
        ("tutorial/first-steps.md", "tutorial"),
        ("tutorial/dependencies/index.md", "tutorial"),
        ("advanced/settings.md", "advanced"),
        ("how-to/configure-swagger-ui.md", "how-to"),
        ("reference/background.md", "reference"),
        ("deployment/concepts.md", "deployment"),
        ("learn/index.md", "learn"),
        ("about/index.md", "about"),
        ("resources/index.md", "other"),
        ("index.md", "other"),
        ("async.md", "other"),
        ("tutorial.md", "other"),
    ],
)
def test_doc_type_for(rel_path: str, doc_type: str) -> None:
    assert doc_type_for(rel_path) == doc_type


@pytest.mark.parametrize(
    ("rel_path", "excluded"),
    [
        ("release-notes.md", True),
        ("fastapi-people.md", True),
        ("css/custom.css", True),
        ("js/x/y.js", True),
        ("tutorial/release-notes.md", False),
        ("tutorial/first-steps.md", False),
    ],
)
def test_is_excluded(configs_dir: Path, rel_path: str, excluded: bool) -> None:
    assert is_excluded(rel_path, load_corpus_config(configs_dir).fastapi) is excluded


def _commit(repo: Path, files: dict[str, str], when: str) -> None:
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    env = {**os.environ, "GIT_COMMITTER_DATE": when, "GIT_AUTHOR_DATE": when}
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false"]
    subprocess.run([*git, "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run([*git, "commit", "-qm", "c"], cwd=repo, check=True, env=env)


def test_git_updated_at_takes_latest_commit_per_file(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _commit(tmp_path, {"docs/a.md": "1", "docs/b.md": "1", "other.md": "1"}, "2026-01-01T10:00:00Z")
    _commit(tmp_path, {"docs/a.md": "2"}, "2026-03-05T12:30:00+02:00")
    _commit(tmp_path, {"other.md": "2"}, "2026-04-01T00:00:00Z")
    updated = git_updated_at(tmp_path, "docs")
    assert updated == {
        "docs/a.md": datetime.fromisoformat("2026-03-05T12:30:00+02:00"),
        "docs/b.md": datetime.fromisoformat("2026-01-01T10:00:00+00:00"),
    }
    # Same answer as the per-file command in PROJECT_SPEC §3.1.
    per_file = subprocess.run(
        ["git", "log", "-1", "--format=%cI", "--", "docs/a.md"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert updated["docs/a.md"] == datetime.fromisoformat(per_file)


def test_git_updated_at_fails_outside_a_repo(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="git log failed"):
        git_updated_at(tmp_path, "docs")
