from docseek.ingest.chunking import chunk_sections
from docseek.ingest.strategies import fixed_chunks, regex_offsets, with_context_header
from docseek.ingest.tokens import RegexTokenCounter
from docseek.testing.fixtures import mini_ingest


def test_fixed_windows_cover_documents_and_map_to_sections() -> None:
    sections = mini_ingest().sections
    chunks = fixed_chunks(sections, regex_offsets, size=40, overlap=8)
    ids = {s.section_id for s in sections}
    assert chunks and all(c.section_id in ids for c in chunks)
    assert all(c.n_tokens <= 40 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # Windows overlap: consecutive windows of one long doc share their boundary tokens.
    doc = max(
        {s.doc_id for s in sections},
        key=lambda d: sum(len(s.text) for s in sections if s.doc_id == d),
    )
    same = [c for c in chunks if c.metadata["doc_id"] == doc]
    assert len(same) >= 2
    tail = regex_offsets(same[0].text)[-8:]
    first_tail = same[0].text[tail[0][0] :]
    assert same[1].text.startswith(first_tail[: len(first_tail) // 2])


def test_context_header_changes_only_embedding_text() -> None:
    sections = mini_ingest().sections
    base = chunk_sections(sections, RegexTokenCounter())
    ctx = with_context_header(base, sections)
    assert [c.text for c in ctx] == [c.text for c in base]
    assert all(c.strategy == "section_ctx" and "::section_ctx::" in c.chunk_id for c in ctx)
    assert ctx[0].text_for_embedding.endswith(base[0].text)
    assert (
        " > ".join(next(s for s in sections if s.section_id == ctx[0].section_id).section_path)
        in ctx[0].text_for_embedding
    )
