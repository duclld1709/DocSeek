import pytest

from docseek.core.ids import (
    doc_id_of_section,
    make_chunk_id,
    make_doc_id,
    make_section_id,
    section_id_of_chunk,
)


def test_ids_follow_spec_format() -> None:
    doc_id = make_doc_id("fastapi-docs", "tutorial/dependencies/index.md")
    section_id = make_section_id(doc_id, "first-steps")
    chunk_id = make_chunk_id(section_id, "section", 0)

    assert doc_id == "fastapi-docs:tutorial/dependencies/index.md"
    assert section_id == "fastapi-docs:tutorial/dependencies/index.md#first-steps"
    assert chunk_id == "fastapi-docs:tutorial/dependencies/index.md#first-steps::section::0"


def test_doc_id_normalises_windows_separators() -> None:
    assert make_doc_id("fastapi-docs", r"tutorial\index.md") == "fastapi-docs:tutorial/index.md"


@pytest.mark.parametrize(
    "path", ["", "/abs/file.md", "../escape.md", "a/../b.md", "a//b.md", "C:/x.md", "a.md#x"]
)
def test_doc_id_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError):
        make_doc_id("fastapi-docs", path)


@pytest.mark.parametrize("source", ["", "FastAPI", "fast api", "a:b"])
def test_doc_id_rejects_bad_source(source: str) -> None:
    with pytest.raises(ValueError):
        make_doc_id(source, "index.md")


@pytest.mark.parametrize("anchor", ["", "a#b", "a::b", "has space"])
def test_section_id_rejects_bad_anchor(anchor: str) -> None:
    with pytest.raises(ValueError):
        make_section_id("fastapi-docs:index.md", anchor)


def test_chunk_id_rejects_bad_parts() -> None:
    with pytest.raises(ValueError):
        make_chunk_id("no-anchor", "section", 0)
    with pytest.raises(ValueError):
        make_chunk_id("d:x.md#a", "fixed::256", 0)
    with pytest.raises(ValueError):
        make_chunk_id("d:x.md#a", "section", -1)


def test_round_trip_chunk_to_section_to_doc() -> None:
    section_id = "internal-kb:runbooks/kb-0012.md#rotate-keys"
    chunk_id = make_chunk_id(section_id, "section+context_header", 3)

    assert section_id_of_chunk(chunk_id) == section_id
    assert doc_id_of_section(section_id) == "internal-kb:runbooks/kb-0012.md"


@pytest.mark.parametrize("chunk_id", ["d:x.md#a", "d:x.md#a::section", "nohash::s::0"])
def test_section_id_of_chunk_rejects_malformed(chunk_id: str) -> None:
    with pytest.raises(ValueError):
        section_id_of_chunk(chunk_id)


def test_doc_id_of_section_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        doc_id_of_section("no-hash-here")
