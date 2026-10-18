"""Markdown structure: headings, anchors, fences, front matter, paths and provenance."""

from pathlib import Path

import pytest

from docseek.ingest.includes import CodeIncludeResolver
from docseek.ingest.markdown_parser import (
    FenceTracker,
    PageMeta,
    clean_title,
    page_path_prefix,
    parse_page,
    slugify,
)

MINI_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "mini_corpus"
MINI_DOCS = MINI_ROOT / "docs" / "en" / "docs"
META = PageMeta(source="fastapi-docs", doc_type="tutorial", acl=("public",))


def _parse(text: str, rel_path: str = "tutorial/page.md") -> list:
    page = parse_page(text, rel_path=rel_path, meta=META, provenance_file=f"docs/{rel_path}")
    return page.sections


def _mini(rel_path: str) -> list:
    text = (MINI_DOCS / rel_path).read_text(encoding="utf-8")
    includes = CodeIncludeResolver(repo_root=MINI_ROOT, mkdocs_dir=MINI_ROOT / "docs" / "en")
    page = parse_page(
        text,
        rel_path=rel_path,
        meta=META,
        provenance_file=f"docs/en/docs/{rel_path}",
        includes=includes,
    )
    return page.sections


def test_anchor_suffix_is_split_into_title_and_anchor() -> None:
    (s,) = _parse("## First Steps { #first-steps }\n\nbody\n")
    assert (s.title, s.anchor) == ("First Steps", "first-steps")
    assert s.section_id == "fastapi-docs:tutorial/page.md#first-steps"
    assert s.text == "body"


def test_heading_without_anchor_gets_mkdocs_style_slug() -> None:
    (s,) = _mini("reference/background.md")
    assert s.title == "Background Tasks - BackgroundTasks"
    assert s.anchor == "background-tasks-backgroundtasks"


@pytest.mark.parametrize(
    ("text", "slug"),
    [
        ("Step 1: import `FastAPI`", "step-1-import-fastapi"),
        ("Café & Crème", "cafe-creme"),
        ("  --  ", "section"),
    ],
)
def test_slugify(text: str, slug: str) -> None:
    assert slugify(clean_title(text)) == slug


def test_clean_title_drops_links_tags_and_backticks() -> None:
    raw = "[Swagger](https://swagger.io/) / <b>`OpenAPI`</b> { #swagger-openapi }"
    assert clean_title(raw) == "Swagger / OpenAPI"


def test_hash_lines_inside_fences_are_not_headings() -> None:
    sections = _mini("tutorial/first-steps.md")
    assert [s.anchor for s in sections] == [
        "first-steps",
        "interactive-api-docs",
        "step-1-import-fastapi",
    ]
    assert "# Install FastAPI first" in sections[0].text
    assert "# This line is a comment, not a heading" in sections[2].text


def test_tilde_and_longer_fences() -> None:
    text = "# A\n~~~\n# not a heading\n```\n# still code\n~~~\n````md\n```\n# code\n````\n## B\n"
    assert [s.anchor for s in _parse(text)] == ["a", "b"]


def test_fence_tracker_needs_matching_fence_to_close() -> None:
    f = FenceTracker()
    assert f.feed("```Python") and f.inside and f.info == "Python"
    assert f.feed("~~~") and f.inside
    assert f.feed("```") and not f.inside


def test_duplicate_anchors_are_made_unique() -> None:
    text = "# Page\n## Notes\n## Notes\n## Notes { #notes }\n"
    assert [s.anchor for s in _parse(text)] == ["page", "notes", "notes_1", "notes_2"]


def test_section_path_follows_heading_hierarchy_and_folders() -> None:
    sections = _mini("tutorial/dependencies/index.md")
    paths = {s.anchor: s.section_path for s in sections}
    assert paths["dependencies"] == ["Tutorial", "Dependencies"]
    assert paths["first-steps"] == ["Tutorial", "Dependencies", "First Steps"]
    assert paths["declare-the-dependency-in-the-dependant"] == [
        "Tutorial",
        "Dependencies",
        "First Steps",
        'Declare the dependency, in the "dependant"',
    ]


def test_sibling_after_deeper_heading_pops_the_stack() -> None:
    text = "# A\n## B\n### C\n## D\n"
    assert _parse(text)[-1].section_path == ["Tutorial", "A", "D"]


@pytest.mark.parametrize(
    ("rel_path", "prefix"),
    [
        ("index.md", []),
        ("async.md", []),
        ("tutorial/index.md", []),
        ("tutorial/first-steps.md", ["Tutorial"]),
        ("tutorial/dependencies/index.md", ["Tutorial"]),
        ("how-to/configure-swagger-ui.md", ["How To"]),
    ],
)
def test_page_path_prefix(rel_path: str, prefix: list[str]) -> None:
    assert page_path_prefix(rel_path) == prefix


def test_provenance_line_ranges_cover_the_page() -> None:
    text = "# A { #a }\n\none\n\n## B { #b }\ntwo\n"
    a, b = _parse(text)
    assert (a.provenance.line_start, a.provenance.line_end) == (1, 4)
    assert (b.provenance.line_start, b.provenance.line_end) == (5, 6)
    assert a.provenance.file == "docs/tutorial/page.md"


def test_front_matter_is_skipped_but_line_numbers_are_kept() -> None:
    text = "---\nhide:\n  - navigation\n---\n\n# Release Notes { #release-notes }\n\nx\n"
    (s,) = _parse(text, "release-notes.md")
    assert s.provenance.line_start == 6
    assert "hide" not in s.text


def test_preamble_before_first_heading_joins_first_section() -> None:
    (s,) = _parse("Intro text.\n\n# Title { #title }\nbody\n")
    assert s.provenance.line_start == 1
    assert s.text.startswith("Intro text.")


def test_page_without_headings_becomes_one_section() -> None:
    (s,) = _parse("just text\n", "tutorial/no-heading.md")
    assert (s.title, s.anchor, s.text) == ("No Heading", "no-heading", "just text")


def test_page_metadata_is_copied_to_sections() -> None:
    (s,) = _parse("# A\n")
    assert (s.source, s.doc_type, s.acl, s.modality) == (
        "fastapi-docs",
        "tutorial",
        ["public"],
        "text",
    )


def test_every_mini_corpus_page_parses_with_unique_section_ids() -> None:
    ids: list[str] = []
    for page in sorted(MINI_DOCS.rglob("*.md")):
        ids += [s.section_id for s in _mini(page.relative_to(MINI_DOCS).as_posix())]
    assert len(ids) == len(set(ids)) > 20
