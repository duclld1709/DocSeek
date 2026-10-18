"""Guards the mini_corpus fixture: every documented quirk is present and references are sane.

This is not a parser test (the parser arrives in M1); it only makes sure the fixture keeps
covering what the parser will need to handle.
"""

import re
from pathlib import Path

import pytest

from docseek.core.config import load_corpus_config

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "mini_corpus"
DOCS = ROOT / "docs" / "en" / "docs"
MKDOCS_DIR = ROOT / "docs" / "en"  # code includes resolve relative to this directory

INCLUDE_RE = re.compile(r"\{\* (?P<path>\S+)(?P<opts>[^*]*)\*\}")
RANGE_RE = re.compile(r"(?P<kind>hl|ln)\[(?P<spec>[^\]]+)\]")
LINK_RE = re.compile(r"\]\((?P<target>[^)#:]+\.md)(?:#(?P<anchor>[^)]+))?\)")
ANCHOR_RE = re.compile(r"^#{1,6} .+ \{ #(?P<anchor>[a-z0-9-]+) \}$", re.M)
DANGLING_LINKS = {("tutorial/dependencies/dependencies-with-yield.md", "../handling-errors.md")}


def _md_files() -> list[Path]:
    return sorted(DOCS.rglob("*.md"))


def _all_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in _md_files())


def test_has_about_ten_pages_across_doc_types() -> None:
    assert 10 <= len(_md_files()) <= 15
    top_dirs = {p.relative_to(DOCS).parts[0] for p in _md_files() if p.parent != DOCS}
    assert {"tutorial", "advanced", "how-to", "reference", "deployment"} <= top_dirs


@pytest.mark.parametrize(
    ("quirk", "pattern"),
    [
        ("anchored heading", r"^## .+ \{ #[a-z0-9-]+ \}$"),
        ("heading without anchor", r"^# [^{\n]+`[^`]+`$"),
        ("hash comment inside fence", r"```(?:console|Python)\n# "),
        ("plain include", r"\{\* \S+ \*\}"),
        ("hl include", r"\{\* \S+ hl\[[^\]]+\] \*\}"),
        ("ln+hl include", r"\{\* \S+ ln\[[^\]]+\] hl\[[^\]]+\] \*\}"),
        ("admonition plain", r"^/// tip$"),
        ("admonition titled", r"^/// note \| .+$"),
        ("admonition warning", r"^/// warning$"),
        ("admonition danger", r"^/// danger$"),
        ("tabs", r"^//// tab \| .+$"),
        ("dfn", r"<dfn title=\"[^\"]+\">"),
        ("abbr", r"<abbr title=\"[^\"]+\">"),
        ("img", r"<img src=\"/img/[^\"]+\">"),
        ("termy div", r"<div class=\"termy\">"),
        ("font tag", r"<font color="),
        ("table", r"^\|[-|]+\|$"),
        ("link with anchor", r"\]\([^)]+\.md#[a-z0-9-]+\)"),
        ("mkdocstrings", r"^::: fastapi\.\w+$"),
        ("front matter", r"\A---\nhide:"),
    ],
)
def test_quirk_present(quirk: str, pattern: str) -> None:
    texts = [p.read_text(encoding="utf-8") for p in _md_files()]
    assert any(re.search(pattern, t, re.M) for t in texts), f"missing quirk: {quirk}"


def test_includes_resolve_and_ranges_fit() -> None:
    includes = list(INCLUDE_RE.finditer(_all_text()))
    assert includes
    for m in includes:
        target = (MKDOCS_DIR / m["path"]).resolve()
        assert target.is_file(), m["path"]
        n_lines = len(target.read_text(encoding="utf-8").splitlines())
        for r in RANGE_RE.finditer(m["opts"]):
            for part in r["spec"].split(","):
                lo, _, hi = part.partition(":")
                assert 1 <= int(lo) <= int(hi or lo) <= n_lines, (m.group(0), n_lines)


def test_variant_pairs_exist_for_an_preference() -> None:
    for stem in ("dependencies/tutorial001", "security/tutorial001"):
        assert (ROOT / "docs_src" / f"{stem}_py310.py").is_file()
        assert (ROOT / "docs_src" / f"{stem}_an_py310.py").is_file()


def test_internal_links_resolve_except_declared_dangling() -> None:
    anchors = {
        p.resolve(): set(ANCHOR_RE.findall(p.read_text(encoding="utf-8"))) for p in _md_files()
    }
    dangling = set()
    for page in _md_files():
        for m in LINK_RE.finditer(page.read_text(encoding="utf-8")):
            target = (page.parent / m["target"]).resolve()
            if target not in anchors:
                dangling.add((page.relative_to(DOCS).as_posix(), m["target"]))
                continue
            if m["anchor"]:
                assert m["anchor"] in anchors[target], (page.name, m.group(0))
    assert dangling == DANGLING_LINKS


def test_excluded_entries_from_corpus_config_exist(configs_dir: Path) -> None:
    cfg = load_corpus_config(configs_dir).fastapi
    present_files = {p.name for p in DOCS.iterdir() if p.is_file()}
    present_dirs = {p.name for p in DOCS.iterdir() if p.is_dir()}
    assert {"release-notes.md", "fastapi-people.md"} <= present_files & set(cfg.exclude_files)
    assert "css" in present_dirs & set(cfg.exclude_dirs)
