"""Internal KB: front matter validation, cross-document checks and section metadata."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from docseek.core.config import load_corpus_config, load_principals
from docseek.ingest.internal_kb import (
    KBValidationError,
    load_kb,
    parse_kb,
    superseded_by,
)

REPO = Path(__file__).resolve().parents[3]
MINI_KB = REPO / "tests" / "fixtures" / "mini_kb"
CONFIGS = REPO / "configs"
KB_CFG = load_corpus_config(CONFIGS).internal_kb
PRINCIPALS = load_principals(CONFIGS)


def _copy(tmp_path: Path) -> Path:
    dest = tmp_path / "kb"
    shutil.copytree(MINI_KB, dest)
    return dest


def _edit(path: Path, old: str, new: str) -> None:
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


def test_mini_kb_loads_and_parses_with_metadata() -> None:
    docs = load_kb(MINI_KB, KB_CFG, PRINCIPALS)
    assert [d.front.id for d in docs] == ["kb-0001", "kb-0002", "kb-0003", "kb-0004"]
    parsed = parse_kb(docs, KB_CFG, REPO)
    assert parsed.issues == []
    by_id = {s.section_id: s for s in parsed.sections}
    new = by_id["internal-kb:kb-0003.md#payment-api-timeouts"]
    assert (new.source, new.doc_type, new.acl, new.version, new.status) == (
        "internal-kb",
        "runbook",
        ["eng-internal"],
        2,
        "active",
    )
    assert new.supersedes == "internal-kb:kb-0002.md"
    assert new.updated_at == datetime(2026, 8, 2, tzinfo=UTC)
    assert new.provenance.file == "tests/fixtures/mini_kb/kb-0003.md"
    assert new.provenance.line_start == 13  # after 11 lines of front matter + blank line
    assert "id: kb-0003" not in new.text
    old = by_id["internal-kb:kb-0002.md#payment-api-timeouts"]
    assert (old.status, old.supersedes) == ("deprecated", None)
    assert by_id["internal-kb:kb-0004.md#jwt-signing-key-rotation"].acl == ["security-team"]


def test_superseded_by() -> None:
    assert superseded_by(load_kb(MINI_KB, KB_CFG, PRINCIPALS)) == {"kb-0002": "kb-0003"}


@pytest.mark.parametrize(
    ("file", "old", "new", "error"),
    [
        ("kb-0001.md", 'acl: ["public"]', 'acl: ["admins"]', "unknown ACL groups"),
        ("kb-0001.md", "doc_type: onboarding", "doc_type: memo", "unknown doc_type"),
        ("kb-0001.md", "status: active", "status: archived", "status"),
        ("kb-0001.md", "version: 1", "version: 0", "version"),
        ("kb-0001.md", "id: kb-0001", "id: kb-1", "kb-0012"),
        ("kb-0001.md", "id: kb-0001", "id: kb-0002", "file name must be"),
        ("kb-0003.md", "supersedes: kb-0002", "supersedes: kb-0099", "supersedes missing"),
        ("kb-0003.md", "supersedes: kb-0002", "supersedes: kb-0003", "itself"),
        ("kb-0003.md", "updated_at: 2026-08-02", "updated_at: 2026-01-01", "must be newer"),
        ("kb-0001.md", "owner: platform-team", "owner: platform-team\nextra: 1", "extra"),
    ],
)
def test_invalid_kb_is_rejected(tmp_path: Path, file: str, old: str, new: str, error: str) -> None:
    kb = _copy(tmp_path)
    _edit(kb / file, old, new)
    with pytest.raises(KBValidationError, match=error):
        load_kb(kb, KB_CFG, PRINCIPALS)


def test_supersedes_cycle_is_rejected_by_date_order(tmp_path: Path) -> None:
    kb = _copy(tmp_path)
    _edit(kb / "kb-0002.md", "supersedes: null", "supersedes: kb-0003")
    _edit(kb / "kb-0002.md", "updated_at: 2026-03-01", "updated_at: 2026-09-01")
    with pytest.raises(KBValidationError, match="must be newer"):
        load_kb(kb, KB_CFG, PRINCIPALS)


def test_missing_front_matter_is_rejected(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "kb-0001.md").write_text("# No front matter\n", encoding="utf-8")
    with pytest.raises(KBValidationError, match="front matter"):
        load_kb(kb, KB_CFG, PRINCIPALS)
