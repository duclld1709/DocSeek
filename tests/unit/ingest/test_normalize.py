"""Normalisation of admonitions, tabs, HTML, mkdocstrings and links (one test per quirk)."""

from pathlib import Path

import pytest

from docseek.ingest.includes import CodeIncludeResolver
from docseek.ingest.markdown_parser import PageMeta, parse_page
from docseek.ingest.normalize import (
    admonition_line,
    extract_links,
    is_console_fence,
    normalize_inline,
)

MINI_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "mini_corpus"
MINI_DOCS = MINI_ROOT / "docs" / "en" / "docs"
META = PageMeta(source="fastapi-docs", doc_type="tutorial", acl=("public",))
SRC = "fastapi-docs"


def _sections(text: str, rel_path: str = "tutorial/p.md") -> list:
    return parse_page(text, rel_path=rel_path, meta=META, provenance_file=rel_path).sections


def _mini_text(rel_path: str) -> str:
    resolver = CodeIncludeResolver(repo_root=MINI_ROOT, mkdocs_dir=MINI_ROOT / "docs" / "en")
    text = (MINI_DOCS / rel_path).read_text(encoding="utf-8")
    page = parse_page(
        text, rel_path=rel_path, meta=META, provenance_file=rel_path, includes=resolver
    )
    return "\n".join(s.text for s in page.sections)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("/// tip", "Tip:"),
        ("/// note | Technical Details", "Note: Technical Details"),
        ("/// details | Preview", "Details: Preview"),
        ("//// tab | Windows PowerShell", "Tab: Windows PowerShell"),
        ("    /// note", "    Note:"),
        ("///", ""),
        ("////", ""),
        ("plain text", None),
        ("// comment", None),
    ],
)
def test_admonition_line(line: str, expected: str | None) -> None:
    assert admonition_line(line) == expected


def test_admonition_body_is_kept_and_markers_removed() -> None:
    (s,) = _sections("# A\n/// warning | Careful\n\nDo **not** do this.\n\n///\n")
    assert s.text == "Warning: Careful\n\nDo **not** do this."


def test_tabs_are_flattened_with_labels() -> None:
    text = _mini_text("advanced/settings.md")
    assert "Tab: Linux, macOS, Windows Bash" in text
    assert "Tab: Windows PowerShell" in text
    assert "////" not in text and "///" not in text


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            'a <dfn title="also known as DI">Dependency Injection</dfn> system',
            "a Dependency Injection (also known as DI) system",
        ),
        (
            '<abbr title="Application Programming Interface">API</abbr>',
            "API (Application Programming Interface)",
        ),
        (
            '<img src="/img/index/index-01-swagger-ui-simple.png">',
            "[image: index-01-swagger-ui-simple.png]",
        ),
        ('<img alt="Swagger UI" src="/img/a.png"/>', "[image: Swagger UI]"),
        ('Go to <a href="http://x" class="external-link">the docs</a>.', "Go to the docs."),
        ("<b>bold</b> and <u>under</u>", "bold and under"),
        ("Tom &amp; Jerry", "Tom & Jerry"),
        ("See [the tutorial](../tutorial/index.md#intro) now", "See the tutorial now"),
        ("![Swagger](img/swagger.png)", "[image: Swagger]"),
        ("keep `<dfn>` and `[x](y.md)` inside code", "keep `<dfn>` and `[x](y.md)` inside code"),
        ("x < y and y > z", "x < y and y > z"),
        ("use [`orjson`](https://github.com/ijl/orjson) here", "use `orjson` here"),
        ("[with `yield` and `scope`](../a.md#b)", "with `yield` and `scope`"),
    ],
)
def test_normalize_inline(line: str, expected: str) -> None:
    assert normalize_inline(line) == expected


def test_termy_div_and_console_font_tags_are_removed() -> None:
    text = _mini_text("tutorial/first-steps.md")
    assert "<div" not in text and "</div>" not in text
    assert "<font" not in text
    assert "$ fastapi dev main.py" in text


def test_html_inside_python_fences_is_untouched() -> None:
    (s,) = _sections('# A\n```Python\nhtml = "<b>hi</b>"\n```\n')
    assert '"<b>hi</b>"' in s.text


@pytest.mark.parametrize(
    ("info", "console"),
    [
        ("console", True),
        ("bash", True),
        ("Python", False),
        ("{ .dockerfile .annotate }", False),
        ("", False),
    ],
)
def test_is_console_fence(info: str, console: bool) -> None:
    assert is_console_fence(info) is console


def test_mkdocstrings_directive_and_options_block() -> None:
    text = (
        "# FastAPI class\n\n::: fastapi.FastAPI\n"
        "    options:\n        members:\n            - get\n\nAfter.\n"
    )
    (s,) = _sections(text, "reference/fastapi.md")
    assert s.text == "API reference: fastapi.FastAPI\n\nAfter."
    assert "API reference: fastapi.BackgroundTasks" in _mini_text("reference/background.md")


@pytest.mark.parametrize(
    ("line", "rel_path", "expected"),
    [
        (
            "[x](dependencies-with-yield.md)",
            "tutorial/dependencies/index.md",
            [f"{SRC}:tutorial/dependencies/dependencies-with-yield.md"],
        ),
        (
            "[x](../path-params.md#order-matters)",
            "tutorial/dependencies/index.md",
            [f"{SRC}:tutorial/path-params.md#order-matters"],
        ),
        (
            "[x](#create-a-dependency)",
            "tutorial/dependencies/index.md",
            [f"{SRC}:tutorial/dependencies/index.md#create-a-dependency"],
        ),
        ("[x](/advanced/settings.md)", "tutorial/a.md", [f"{SRC}:advanced/settings.md"]),
        ("[x](https://fastapi.tiangolo.com/a.md)", "a.md", []),
        ("[x](mailto:a@b.c)", "a.md", []),
        ("[x](../../outside.md)", "tutorial/a.md", []),
        ("[x](../img/a.png)", "tutorial/a.md", []),
        ("`[x](a.md)` is code", "tutorial/a.md", []),
        ("[a](a.md) and [b](a.md) again", "x.md", [f"{SRC}:a.md"]),
        (
            "[`Depends`](tutorial/dependencies/index.md)",
            "x.md",
            [f"{SRC}:tutorial/dependencies/index.md"],
        ),
        ("![diagram](diagram.md)", "x.md", []),
    ],
)
def test_extract_links(line: str, rel_path: str, expected: list[str]) -> None:
    assert extract_links(line, rel_path=rel_path, source=SRC) == expected


def test_links_are_stored_on_sections_without_self_links() -> None:
    text = "# A { #a }\nSee [B](#b), [me](#a) and [Q](query-params.md).\n## B { #b }\n"
    a, _ = _sections(text)
    assert a.links == [f"{SRC}:tutorial/p.md#b", f"{SRC}:tutorial/query-params.md"]
    assert a.text == "See B, me and Q."


def test_links_inside_fences_are_ignored() -> None:
    (s,) = _sections("# A\n```md\n[x](other.md)\n```\n")
    assert s.links == []


def test_no_raw_markers_left_in_mini_corpus() -> None:
    for path in sorted(MINI_DOCS.rglob("*.md")):
        text = _mini_text(path.relative_to(MINI_DOCS).as_posix())
        for marker in ("<dfn", "<abbr", "<img", "<div", "///", ":::", "](", "{ #"):
            assert marker not in text, (path.name, marker)
