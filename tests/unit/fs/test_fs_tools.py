"""FS tools: sandbox (path traversal, symlinks), ACL hiding, index filtering, limits."""

import os
from pathlib import Path

import pytest

from docseek.core.acl import AccessPolicy
from docseek.core.config import UnknownPrincipalError, load_principals
from docseek.core.tracing import MemoryTraceSink, Tracer
from docseek.fs.tools import FsError, FsTools
from docseek.ingest.projection import project
from docseek.testing.fixtures import REPO_ROOT, mini_ingest

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))
SECRET_FILE = "internal-kb/kb-0004.md"  # security-team only


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("projected")
    project(mini_ingest().sections, out)
    return out


@pytest.mark.parametrize(
    "bad",
    [
        "../secrets.txt",
        "fastapi-docs/../../x",
        r"..\..\Windows\win.ini",
        "/etc/passwd",
        "C:/Windows/win.ini",
        r"c:\x",
        "internal-kb/../../../x.md",
    ],
)
def test_path_traversal_is_rejected(root: Path, bad: str) -> None:
    fs = FsTools(root, POLICY, "security")
    for call in (lambda: fs.cat(bad), lambda: fs.ls(bad), lambda: fs.grep("x", bad)):
        with pytest.raises(FsError, match="not allowed"):
            call()


def test_symlinks_are_rejected(root: Path, tmp_path: Path) -> None:
    target = tmp_path / "outside.md"
    target.write_text("outside", encoding="utf-8")
    link = root / "fastapi-docs" / "link.md"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlinks need extra privileges on this OS")
    try:
        with pytest.raises(FsError, match="not allowed"):
            FsTools(root, POLICY, "security").cat("fastapi-docs/link.md")
    finally:
        link.unlink()


def test_acl_hides_restricted_files_like_missing_ones(root: Path) -> None:
    anon, sec = FsTools(root, POLICY, "anonymous"), FsTools(root, POLICY, "security")
    assert "JWT" in sec.cat(SECRET_FILE)
    with pytest.raises(FsError) as hidden:
        anon.cat(SECRET_FILE)
    with pytest.raises(FsError) as missing:
        anon.cat("internal-kb/kb-9999.md")
    assert str(hidden.value).replace("kb-0004", "X") == str(missing.value).replace("kb-9999", "X")
    assert "kb-0004.md" not in anon.ls("internal-kb") and "kb-0004.md" in sec.ls("internal-kb")
    assert SECRET_FILE not in anon.find("**/*.md") and SECRET_FILE in sec.find("*/*.md")
    assert not anon.grep("signing key", ".") and sec.grep("signing key", ".")
    index = anon.cat("internal-kb/_index.md")
    assert "kb-0004" not in index and "JWT" not in index and "kb-0001" in index
    assert "_manifest.json" not in anon.ls(".")
    with pytest.raises(FsError):
        anon.cat("_manifest.json")
    with pytest.raises(UnknownPrincipalError):
        FsTools(root, POLICY, "intruder")


def test_cat_by_section_and_limits(root: Path) -> None:
    sink = MemoryTraceSink()
    fs = FsTools(root, POLICY, "engineer", tracer=Tracer(sink))
    sid = "internal-kb:kb-0003.md#"
    page = fs.cat("internal-kb/kb-0003.md")
    marker = next(ln for ln in page.splitlines() if "section_id:" in ln)
    first_sid = marker.split("section_id: ")[1]
    part = fs.cat("internal-kb/kb-0003.md", section_id=first_sid.replace(" -->", ""))
    assert sid in part and len(part) < len(page) + 1
    assert len(fs.grep("e", ".", max_results=500)) == 50
    with pytest.raises(FsError, match="too long"):
        fs.grep("a" * 500)
    with pytest.raises(FsError, match="bad regex"):
        fs.grep("(", regex=True)
    assert {e.name for e in sink.events} >= {"fs.cat", "fs.grep"}
