"""`section` chunking: size limit, paragraph overlap, code fences and payload."""

import itertools
from datetime import UTC, datetime

import pytest

from docseek.core.ids import section_id_of_chunk
from docseek.core.models import Provenance, Section
from docseek.ingest.chunking import ChunkingConfig, chunk_section, chunk_sections
from docseek.ingest.markdown_parser import FenceTracker
from docseek.ingest.tokens import RegexTokenCounter

COUNTER = RegexTokenCounter()


def _section(text: str, anchor: str = "s") -> Section:
    doc_id = "fastapi-docs:tutorial/p.md"
    return Section(
        section_id=f"{doc_id}#{anchor}",
        doc_id=doc_id,
        source="fastapi-docs",
        title="Title",
        anchor=anchor,
        section_path=["Tutorial", "Title"],
        text=text,
        doc_type="tutorial",
        acl=["public"],
        updated_at=datetime(2026, 7, 24, tzinfo=UTC),
        provenance=Provenance(file="docs/p.md", line_start=1, line_end=9),
    )


def _para(i: int, words: int = 30) -> str:
    return " ".join(f"p{i}w{j}" for j in range(words))


def test_short_section_is_one_chunk_with_flat_metadata() -> None:
    (c,) = chunk_section(_section("Hello `Depends` world."), COUNTER)
    assert c.chunk_id == "fastapi-docs:tutorial/p.md#s::section::0"
    assert section_id_of_chunk(c.chunk_id) == c.section_id
    assert c.text == c.text_for_embedding == "Hello `Depends` world."
    assert c.n_tokens == COUNTER.count(c.text)
    assert c.metadata == {
        "section_id": "fastapi-docs:tutorial/p.md#s",
        "doc_id": "fastapi-docs:tutorial/p.md",
        "source": "fastapi-docs",
        "title": "Title",
        "section_path": "Tutorial > Title",
        "doc_type": "tutorial",
        "modality": "text",
        "acl": ["public"],
        "status": "active",
        "version": None,
        "supersedes": None,
        "updated_at": "2026-07-24T00:00:00+00:00",
    }


def test_empty_section_has_no_chunks() -> None:
    assert chunk_section(_section("  \n"), COUNTER) == []


def test_long_section_splits_on_paragraphs_within_limit_and_overlaps() -> None:
    paras = [_para(i, 30) for i in range(10)]  # 30 tokens each
    cfg = ChunkingConfig(max_tokens=100, overlap_tokens=30)
    chunks = chunk_section(_section("\n\n".join(paras)), COUNTER, cfg)
    assert len(chunks) > 1
    assert [c.chunk_id.rsplit("::", 1)[1] for c in chunks] == [str(i) for i in range(len(chunks))]
    for c in chunks:
        assert c.n_tokens <= 100
        assert all(p in paras for p in c.text.split("\n\n"))  # never cuts a paragraph
    for prev, nxt in itertools.pairwise(chunks):
        assert prev.text.split("\n\n")[-1] == nxt.text.split("\n\n")[0]  # one-paragraph overlap
    covered = {p for c in chunks for p in c.text.split("\n\n")}
    assert covered == set(paras)


def test_zero_overlap_partitions_paragraphs() -> None:
    paras = [_para(i, 30) for i in range(7)]
    cfg = ChunkingConfig(max_tokens=100, overlap_tokens=0)
    chunks = chunk_section(_section("\n\n".join(paras)), COUNTER, cfg)
    assert [p for c in chunks for p in c.text.split("\n\n")] == paras


def test_code_block_with_blank_lines_stays_whole_when_it_fits() -> None:
    code = "```python\nimport a\n\n\ndef f():\n    return 1\n```"
    text = "\n\n".join([_para(0, 60), code, _para(1, 60)])
    chunks = chunk_section(_section(text), COUNTER, ChunkingConfig(100, 0))
    assert any(code in c.text for c in chunks)


def test_oversized_code_block_is_split_and_refenced() -> None:
    body = "\n".join(f"x{i} = call_{i}(a, b)" for i in range(80))
    text = f"Intro.\n\n```python\n{body}\n```"
    chunks = chunk_section(_section(text), COUNTER, ChunkingConfig(100, 10))
    assert len(chunks) > 2
    for c in chunks:
        assert c.n_tokens <= 100
        fences = FenceTracker()
        for line in c.text.splitlines():
            fences.feed(line)
        assert not fences.inside, c.text  # every piece closes its fence
    lines = [ln for c in chunks for ln in c.text.splitlines() if ln.startswith("x")]
    assert sorted(set(lines), key=lambda s: int(s.split()[0][1:])) == body.splitlines()


def test_single_huge_line_is_split_by_words() -> None:
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_section(_section(text), COUNTER, ChunkingConfig(100, 0))
    assert all(c.n_tokens <= 100 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == text.split()


def test_big_block_after_small_one_does_not_leave_a_subsumed_chunk() -> None:
    text = "\n\n".join([_para(0, 5), _para(1, 90)])
    chunks = chunk_section(_section(text), COUNTER, ChunkingConfig(100, 10))
    assert [c.text for c in chunks] == [text]


def test_chunk_sections_flattens_in_order() -> None:
    chunks = chunk_sections([_section("a", "a"), _section("", "b"), _section("c", "c")], COUNTER)
    assert [c.section_id.rsplit("#", 1)[1] for c in chunks] == ["a", "c"]


@pytest.mark.parametrize(("max_tokens", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_config(max_tokens: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(max_tokens, overlap)
