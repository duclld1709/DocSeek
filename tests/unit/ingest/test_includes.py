"""Code includes: resolution root, variants, line ranges, titles and failure reporting."""

from pathlib import Path

import pytest

from docseek.ingest.includes import (
    CodeIncludeResolver,
    best_variant,
    find_unresolved_includes,
    select_lines,
)
from docseek.ingest.markdown_parser import PageMeta, parse_page

MINI_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "mini_corpus"
MINI_DOCS = MINI_ROOT / "docs" / "en" / "docs"
META = PageMeta(source="fastapi-docs", doc_type="tutorial", acl=("public",))


@pytest.fixture
def resolver() -> CodeIncludeResolver:
    return CodeIncludeResolver(repo_root=MINI_ROOT, mkdocs_dir=MINI_ROOT / "docs" / "en")


def _parse(text: str, resolver: CodeIncludeResolver | None, rel_path: str = "tutorial/p.md"):
    return parse_page(
        text, rel_path=rel_path, meta=META, provenance_file=rel_path, includes=resolver
    )


def test_include_resolves_relative_to_mkdocs_dir(resolver: CodeIncludeResolver) -> None:
    page = _parse("# A\n{* ../../docs_src/first_steps/tutorial001_py310.py *}\n", resolver)
    (s,) = page.sections
    code = (MINI_ROOT / "docs_src/first_steps/tutorial001_py310.py").read_text(encoding="utf-8")
    assert s.text == "```python\n" + code.rstrip("\n") + "\n```"
    assert s.code_refs == ["docs_src/first_steps/tutorial001_py310.py"]
    assert page.issues == []


def test_hl_is_ignored_and_ln_selects_lines(resolver: CodeIncludeResolver) -> None:
    path = "../../docs_src/dependencies/tutorial007_an_py310.py"
    full = (MINI_ROOT / path.removeprefix("../../")).read_text(encoding="utf-8").splitlines()
    (s,) = _parse(f"# A\n{{* {path} ln[2:3] hl[3] *}}\n", resolver).sections
    assert s.text.splitlines()[1:-1] == full[1:3]


def test_explicit_variant_is_kept_even_if_an_variant_exists(
    resolver: CodeIncludeResolver,
) -> None:
    text = "# A\n{* ../../docs_src/dependencies/tutorial001_py310.py *}\n"
    (s,) = _parse(text, resolver).sections
    assert s.code_refs == ["docs_src/dependencies/tutorial001_py310.py"]


def test_missing_file_falls_back_to_an_py310_and_drops_ln(
    resolver: CodeIncludeResolver,
) -> None:
    page = _parse("# A\n{* ../../docs_src/dependencies/tutorial001_py39.py ln[1:2] *}\n", resolver)
    (s,) = page.sections
    assert s.code_refs == ["docs_src/dependencies/tutorial001_an_py310.py"]
    an = (MINI_ROOT / s.code_refs[0]).read_text(encoding="utf-8").splitlines()
    assert s.text.splitlines()[1:-1] == an
    assert [i.kind for i in page.issues] == ["include-variant-fallback"]


def test_unavailable_file_becomes_placeholder_and_issue(resolver: CodeIncludeResolver) -> None:
    page = _parse("# A\n{* ../../fastapi/openapi/docs.py ln[9:24] hl[18:24] *}\n", resolver)
    (s,) = page.sections
    assert s.text == "(code not available in this corpus: fastapi/openapi/docs.py)"
    assert s.code_refs == ["fastapi/openapi/docs.py"]
    assert [(i.kind, i.line) for i in page.issues] == [("include-missing", 2)]


def test_include_escaping_repo_is_refused(resolver: CodeIncludeResolver) -> None:
    page = _parse("# A\n{* ../../../../../../etc/passwd *}\n", resolver)
    assert page.sections[0].code_refs == []
    assert [i.kind for i in page.issues] == ["include-outside"]


def test_title_option_adds_file_header(resolver: CodeIncludeResolver) -> None:
    text = '# A\n{* ../../docs_src/first_steps/tutorial001_py310.py title["app/main.py"] *}\n'
    (s,) = _parse(text, resolver).sections
    assert s.text.startswith("File: app/main.py\n```python\n")


def test_include_inside_fence_is_left_alone(resolver: CodeIncludeResolver) -> None:
    text = "# A\n```md\n{* ../../docs_src/x.py *}\n```\n"
    page = _parse(text, resolver)
    assert "{* ../../docs_src/x.py *}" in page.sections[0].text
    assert page.issues == []


def test_without_resolver_includes_are_reported() -> None:
    page = _parse("# A\n{* ../../docs_src/x.py hl[1] *}\n", None)
    assert "{*" not in page.sections[0].text
    assert [i.kind for i in page.issues] == ["include-unresolved"]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("2", ["b"]),
        ("2:3", ["b", "c"]),
        ("1,3:4", ["a", "# ...", "c", "d"]),
        ("1:2,3", ["a", "b", "c"]),
        ("4:99", ["d", "e"]),
        ("9:12", []),
    ],
)
def test_select_lines(spec: str, expected: list[str]) -> None:
    assert select_lines(["a", "b", "c", "d", "e"], spec) == expected


def test_best_variant_prefers_an_py310(tmp_path: Path) -> None:
    for name in ("t001_py310.py", "t001_an_py310.py", "t001_py39.py"):
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    assert best_variant(tmp_path / "t001_an_py39.py") == tmp_path / "t001_an_py310.py"
    assert best_variant(tmp_path / "t002_py310.py") is None


def test_no_include_syntax_left_in_mini_corpus(resolver: CodeIncludeResolver) -> None:
    n_refs = 0
    for path in sorted(MINI_DOCS.rglob("*.md")):
        rel = path.relative_to(MINI_DOCS).as_posix()
        page = _parse(path.read_text(encoding="utf-8"), resolver, rel)
        assert page.issues == [], rel
        for s in page.sections:
            assert find_unresolved_includes(s.text) == [], s.section_id
            n_refs += len(s.code_refs)
    assert n_refs >= 7


def test_find_unresolved_includes_ignores_dict_unpacking() -> None:
    text = 'x = {**old, "k": 1}\n{* ../../docs_src/a.py hl[1] *}\nf({*args})'
    assert find_unresolved_includes(text) == ["{* ../../docs_src/a.py hl[1] *}"]
