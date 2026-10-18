"""Leakage@k ACL map: chunk-only sections (scan pages) use their chunk ACL, parsed win."""

from docseek.core.models import Chunk
from docseek.eval.runner import section_acls
from docseek.testing.fixtures import mini_ingest


def _chunk(section_id: str, acl: list[str]) -> Chunk:
    return Chunk(
        chunk_id=f"{section_id}::scan::0",
        section_id=section_id,
        strategy="scan",
        text="t",
        text_for_embedding="t",
        metadata={"acl": acl},
        n_tokens=1,
    )


def test_section_acls_include_chunk_only_sections() -> None:
    ingest = mini_ingest()
    first = ingest.sections[0]
    acl = section_acls(
        ingest.sections,
        [_chunk("scan:x.pdf#page-1", ["security-team"]), _chunk(first.section_id, ["public"])],
    )
    assert acl["scan:x.pdf#page-1"] == ["security-team"]
    assert acl[first.section_id] == list(first.acl)  # the parsed section's ACL is kept
    assert "scan:y.pdf#page-1" not in acl  # still unknown -> leakage_at_k fails closed
