"""Image references: src resolution, size/format filter, section ownership and ACL."""

from pathlib import Path

from docseek.core.models import Provenance, Section
from docseek.ingest.images import (
    find_image_refs,
    find_images_in_line,
    image_chunks,
    resolve_src,
)


def _section(anchor: str, lines: tuple[int, int], acl: list[str]) -> Section:
    doc_id = "fastapi-docs:tutorial/p.md"
    return Section(
        section_id=f"{doc_id}#{anchor}",
        doc_id=doc_id,
        source="fastapi-docs",
        title=anchor.title(),
        anchor=anchor,
        section_path=["Tutorial", anchor.title()],
        text="body",
        doc_type="tutorial",
        acl=acl,
        updated_at=None,
        provenance=Provenance(
            file="docs/en/docs/tutorial/p.md", line_start=lines[0], line_end=lines[1]
        ),
    )


def test_find_images_in_line_html_and_markdown() -> None:
    line = '<img src="/img/a.png" alt="Swagger UI"> and ![diagram](../img/b.jpg "t")'
    assert find_images_in_line(line) == [("/img/a.png", "Swagger UI"), ("../img/b.jpg", "diagram")]


def test_resolve_src() -> None:
    md = "docs/en/docs/tutorial/p.md"
    assert resolve_src("/img/a.png", md) == "docs/en/docs/img/a.png"
    assert resolve_src("../img/b.png", md) == "docs/en/docs/img/b.png"
    assert resolve_src("https://x.org/a.png", md) is None
    assert resolve_src("../../../../../etc/x.png", md) is None  # escapes the repo


def test_find_image_refs_filters_and_assigns_sections(tmp_path: Path) -> None:
    img = tmp_path / "docs/en/docs/img"
    img.mkdir(parents=True)
    (img / "big.png").write_bytes(b"\x89PNG" + b"0" * 6000)
    (img / "icon.png").write_bytes(b"\x89PNG" + b"0" * 100)  # too small
    (img / "diag.svg").write_bytes(b"<svg/>" + b"0" * 6000)  # vector, skipped
    md = tmp_path / "docs/en/docs/tutorial/p.md"
    md.parent.mkdir(parents=True)
    md.write_text(
        '# Intro\n<img src="/img/icon.png">\n## Docs\nSee:\n<img src="/img/big.png">\n'
        '<img src="/img/diag.svg">\n<img src="/img/missing.png">\n',
        encoding="utf-8",
    )
    intro, docs = _section("intro", (1, 2), ["public"]), _section("docs", (3, 7), ["staff"])
    refs = find_image_refs([intro, docs], tmp_path)
    assert [(r.section_id, r.path, r.line) for r in refs] == [
        (docs.section_id, "docs/en/docs/img/big.png", 5)
    ]
    (chunk,) = image_chunks(refs, {docs.section_id: docs}, {refs[0].image_id: "A form."}, {})
    assert chunk.chunk_id == f"{docs.section_id}::image::0"
    assert chunk.metadata["acl"] == ["staff"]  # inherited from the owning section
    assert chunk.metadata["modality"] == "image"
    assert chunk.text == "Caption: A form."
